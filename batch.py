from main import process_query_internally

# Read questions from file
with open("questions.txt", "r") as f:
    questions = f.readlines()

# Process each question
for i, question in enumerate(questions, 1):
    question = question.strip()
    if question:
        print(f"\n=== Question {i}: {question} ===")
        answer = process_query_internally(question, show_thinking=False)
        print(f"Answer: {answer}")
