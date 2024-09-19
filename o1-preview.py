import json
from openai import OpenAI
import os
# from dotenv import load_dotenv
from typing import List, Dict, Tuple
from pydantic import BaseModel
from openai import OpenAI

class ARCResponse(BaseModel):
    output: List[List[int]]


class ARCTester:
    def __init__(self, model=None, task_sets=None):
        self.model = model
        self.llm = OpenAI(
            # This is the default and can be omitted
            api_key = "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"
        )
        self.challenges, self.solutions = self.load_tasks_from_file(task_sets)

    def load_tasks_from_file(self, task_sets):
        """
        Loads the tasks from the file and returns the challenges and solutions tasks
        """
        with open(task_sets['challenges'], "r") as tasks:
            challenges = json.load(tasks)

        with open(task_sets['solutions'], "r") as tasks:
            solutions = json.load(tasks)

        return challenges, solutions

    def chat_completion(self, message):
        """
        Chat completion with OpenAI
        """
        chat_completion = self.llm.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            model=self.model,
        )
        return chat_completion.choices[0].message.content

    def load_prompt(self, name):
        return """You are a bot that is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
Identify the pattern in the training examples, then apply that pattern to the test input to give a final output
Just give valid json list of lists response back, nothing else. Do not wrap your response in "json" tags. Just give the list of lists.
{task_string}
Your response:"""

    def initial_task_response(self, task_id, test_input_index) -> List[List]:
        """
        Given a task, predict the test output
        """

        # Get the string representation of your task
        task_string = self.json_task_to_string(self.challenges, task_id, test_input_index)
        system_prompt = self.load_prompt("system_prompt")
        prompt = system_prompt.format(task_string=task_string)
        response = self.chat_completion(prompt)

        return response

    def json_task_to_string(self, challenge_tasks: dict, task_id: str, test_input_index: int) -> str:
        """
        challenge_tasks: dict a list of tasks
        task_id: str the id of the task we want to convert to a string

        Convert your json task into a string so you can pass it to your LLM.
        This is a crucial step where you can use your creativity to edit how tasks are represented.
        """
        json_task = challenge_tasks[task_id]

        final_output = ""

        train_tasks = json_task['train']
        test_task = json_task['test']

        final_output = "Training Examples\n"

        for i, task in enumerate(train_tasks):
            final_output += f"Example {i + 1}: Input\n["
            for row in task['input']:
                final_output += f"\n{str(row)},"

            final_output += "]\n\n"
            final_output += f"Example {i + 1}: Output\n["

            for row in task['output']:
                final_output += f"\n{str(row)},"

            final_output += "]\n\n"

        final_output += "Test\n["
        for row in test_task[test_input_index]['input']:
            final_output += f"\n{str(row)}"

        return final_output

    def extract_json_from_response(self, input_response):
        """
        Extracts the json from the response
        """
        completion = self.llm.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Extract the json from the response."},
                {"role": "user", "content": input_response},
            ],
            response_format=ARCResponse,
        )

        response = completion.choices[0].message.parsed.output

        return response

    def get_task_prediction(self, task_id, test_input_index) -> List[List]:
        """
        challenge_tasks: dict a list of tasks
        task_id: str the id of the task we want to get a prediction for
        test_input_index: the index of your test input. 96% of tests only have 1 input.

        Given a task, predict the test output
        """

        # Get the string representation of your task
        initial_response = self.initial_task_response(task_id, test_input_index)

        try:
            # Try to parse the raw JSON response
            json_response = json.loads(initial_response)
        except json.JSONDecodeError:
            # If raw JSON parsing fails, try to parse as if it were in a code block
            try:
                code_block_start = initial_response.find("```json")
                code_block_end = initial_response.find("```", code_block_start + 6)
                if code_block_start != -1 and code_block_end != -1:
                    json_response = json.loads(initial_response[code_block_start + 6:code_block_end].strip())
                    # print(f"Parsed JSON response from code block: {json_response}")
                else:
                    raise json.JSONDecodeError("No JSON code block found", initial_response, 0)
            except json.JSONDecodeError:
                # If parsing as a code block also fails, use the extract_json_from_response function
                print("Failed to parse raw JSON and JSON code block, using extract_json_from_response function.")
                json_response = self.extract_json_from_response(initial_response)
                print(f"Extracted JSON response: {json_response}")

        return json_response

    def score_task(self, task_id, task_submission) -> float:
        """
        Scores a single task submission against the solutions.
        """
        task_score = 0
        num_pairs = len(task_submission)

        # Go through each task. Most will only have 1
        for pair_index, pair_attempts in enumerate(task_submission):
            pair_correct = False

            # Look at both of your attempts
            for attempt_key, attempt in pair_attempts.items():
                # check to see if one is correct
                if attempt == self.solutions[task_id][pair_index]:
                    #                     print(f"Task Id {task_id} pair {pair_index+1} {attempt_key} matches solution")
                    pair_correct = True
                    break  # If it is correct, log it and break the loop

            if pair_correct:
                task_score += 1

        task_score /= num_pairs
        return task_score

    def score_submission(self, submission_file_name) -> Tuple[float, int]:
        """
        submission_file_name: str, the file name of your submission file
        solutions: dict, the ground truth solutions you'd like to test against

        Read a submission from file, score it, then return the score
        """
        print(f"Scoring {submission_file_name}\n")

        # Open your submission file
        with open(submission_file_name, "r") as file:
            submission = json.load(file)

        total_score = 0
        total_tasks = 0

        # Loop through each task in your submission to grade it
        for task_id, task_submission in submission.items():
            total_tasks += 1
            task_score = self.score_task(task_id, task_submission)
            total_score += task_score

        return {
            'total_score': total_score,
            'total_tasks_scored': total_tasks
        }

    def run_model(self, NUM_ATTEMPTS=2, NUM_TASKS=None):
        """
        Runs the model for task predictions.
        """
        # Load existing submissions if the file exists
        if os.path.exists("submission.json"):
            with open("submission.json", "r") as f:
                submission = json.load(f)
        else:
            submission = {}

        total_score = 0

        # Process tasks sequentially
        for i, task_id in enumerate(self.challenges):
            if task_id in submission:
                continue

            submission[task_id] = []

            for t, pair in enumerate(self.challenges[task_id]['test']):
                submission[task_id].append({})
                for attempt in range(1, NUM_ATTEMPTS + 1):
                    print(f"Starting prediction for Task ID {task_id}, Test Pair #{t}, Attempt #{attempt}")

                    prediction = self.get_task_prediction(task_id, t)
                    attempt_key = f"attempt_{attempt}"

                    submission[task_id][t][attempt_key] = prediction

                    # Save the submission to a file after each prediction
                    with open("submission.json", "w") as f:
                        json.dump(submission, f, indent=4)

            # Score the task
            score = self.score_task(task_id, submission[task_id])
            total_score += score
            print(f"Task {task_id} score: {score}, Total score: {total_score}\n")

            # If you want to stop after N tasks
            if NUM_TASKS is not None and i + 1 == NUM_TASKS:
                break

        return submission

task_sets = {
    'train' : {
        'challenges' : '/Users/yuni/PycharmProjects/pythonProject/arc-agi_training_challenges.json',
        'solutions' : '/Users/yuni/PycharmProjects/pythonProject/arc-agi_training_solutions.json'
    },
    'eval' : {
        'challenges' : '/Users/yuni/PycharmProjects/pythonProject/arc-agi_evaluation_challenges.json',
        'solutions' : '/Users/yuni/PycharmProjects/pythonProject/arc-agi_evaluation_solutions.json'
    }
}

# Swap the openai model you'd like here
t = ARCTester(model="o1-preview", task_sets=task_sets['eval'])
t.run_model(NUM_TASKS=1)
print(t.score_submission("submission.json"))

