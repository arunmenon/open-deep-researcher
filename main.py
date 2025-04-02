import argparse
import asyncio
import os
import time

from src.deep_research import DeepSearch

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


async def main():
    parser = argparse.ArgumentParser(description='Run deep search queries')
    parser.add_argument('query', type=str, help='The search query')
    parser.add_argument('--mode', type=str, choices=['fast', 'balanced', 'comprehensive'],
                      default='balanced', help='Research mode (default: balanced)')
    parser.add_argument('--num-queries', type=int, default=3,
                      help='Number of queries to generate (default: 3)')
    parser.add_argument('--learnings', nargs='*', default=[],
                      help='List of previous learnings')
    parser.add_argument('--runpod-api-key', type=str, default=None,
                      help='RunPod API key (can also be set via RUNPOD_API_KEY environment variable)')
    parser.add_argument('--runpod-endpoint-id', type=str, default=None,
                      help='RunPod endpoint ID (can also be set via RUNPOD_ENDPOINT_ID environment variable)')

    args = parser.parse_args()

    # Start the timer
    start_time = time.time()

    # Get RunPod API key either from args or environment
    runpod_api_key = args.runpod_api_key or os.getenv('RUNPOD_API_KEY')
    if not runpod_api_key:
        raise ValueError("Please set RUNPOD_API_KEY environment variable")
    
    # Get RunPod endpoint ID either from args or environment
    runpod_endpoint_id = args.runpod_endpoint_id or os.getenv('RUNPOD_ENDPOINT_ID')
    
    # Create DeepSearch instance with new modular approach
    deep_search = DeepSearch(
        mode=args.mode,
        runpod_api_key=runpod_api_key,
        runpod_endpoint_id=runpod_endpoint_id
    )

    breadth_and_depth = await deep_search.determine_research_breadth_and_depth(
        args.query)

    # The function returns a BreadthDepthResponse object
    breadth = breadth_and_depth.breadth
    depth = breadth_and_depth.depth
    explanation = breadth_and_depth.explanation

    print(f"Breadth: {breadth}")
    print(f"Depth: {depth}")
    print(f"Explanation: {explanation}")

    # If running as part of a test, use hardcoded answers to avoid input prompt
    print("To better understand your research needs, please answer these follow-up questions:")

    try:
        follow_up_questions = await deep_search.generate_follow_up_questions(args.query)
    except Exception as e:
        print(f"Error generating follow-up questions: {e}")
        follow_up_questions = [
            f"What specific aspects of {args.query} are you interested in?",
            f"What is your goal for researching {args.query}?",
            f"Any specific timeframe or context for {args.query}?"
        ][:3]

    # When running in a test environment, use default answers
    if os.getenv("RUN_MODE") == "test" or len(follow_up_questions) == 0:
        print("Using default answers for follow-up questions in test mode")
        answers = []
        for question in follow_up_questions:
            print(f"{question}: Interested in a comprehensive overview")
            answers.append({
                "question": question,
                "answer": "Interested in a comprehensive overview"
            })
    else:
        # get answers to the follow up questions interactively
        answers = []
        for question in follow_up_questions:
            try:
                answer = input(f"{question}: ")
                answers.append({
                    "question": question,
                    "answer": answer
                })
            except EOFError:
                # If we can't get input (e.g., running in a script), use a default answer
                print(f"{question}: Default answer: Interested in a comprehensive overview")
                answers.append({
                    "question": question,
                    "answer": "Interested in a comprehensive overview"
                })

    questions_and_answers = "\n".join(
        [f"{answer['question']}: {answer['answer']}" for answer in answers])

    combined_query = f"Initial query: {args.query}\n\n Follow up questions and answers: {questions_and_answers}"

    print(f"\nHere is the combined query: {combined_query}\n\n")

    print("Starting research... \n")

    # Run the deep research
    results = await deep_search.deep_research(
        query=combined_query,
        breadth=breadth,
        depth=depth,
        learnings=[],
        visited_urls={}
    )

    # Generate and print the final report
    final_report = await deep_search.generate_final_report(
        query=combined_query,
        learnings=results["learnings"],
        visited_urls=results["visited_urls"]
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print("\nFinal Research Report:")
    print("=====================")
    print(final_report)
    print(f"\nTotal research time: {minutes} minutes and {seconds} seconds")

    # Save the report to a file
    with open("final_report.md", "w") as f:
        f.write(final_report)
        f.write(
            f"\n\nTotal research time: {minutes} minutes and {seconds} seconds")


if __name__ == "__main__":
    asyncio.run(main())