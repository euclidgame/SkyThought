import json
import matplotlib.pyplot as plt
import argparse
import re

def clean_latex(text):
    """Cleans LaTeX expressions and removes unsupported commands for Matplotlib rendering."""
    text = text.replace("\\implies", "\\Rightarrow")
    text = text.replace("\\boxed", "")  # Remove \boxed
    text = re.sub(r'\\let.*', '', text)  # Remove \let commands
    text = re.sub(r'\\newcommand.*', '', text)  # Remove \newcommand definitions
    text = re.sub(r'\\def.*', '', text)  # Remove \def commands
    text = re.sub(r'\\textbf{(.*?)}', r'\\mathbf{\1}', text)  # Convert \textbf{} to \mathbf{}
    text = re.sub(r'\$\$(.*?)\$\$', r'\n $ \1 $ \n', text)  # Convert block math $$...$$ to \n $ ... $ \n
    print(text)
    text = re.sub(r'\\\[(.*?)\\\]', r'$\1$', text, flags=re.S)  # Convert \[ ... \] to \n $ ... $ \n
    print(text)
    text = re.sub(r'\$\n(.*?)\n\$', r'$\1$', text, flags=re.S)
    print(text)
    text = re.sub(r'(?<!\\)\$(.*?)\$', r'$\1$', text)  # Ensure inline math uses $...$
    return text

def visualize_json(json_path, problem_id):
    # Load the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Find the problem with the given ID
    problem_details = None
    for key, value in data.items():
        if value.get("id") == problem_id:
            problem_details = value
            break
    
    if not problem_details:
        print(f"Problem ID {problem_id} not found.")
        return
    
    # Extract and clean LaTeX expressions
    problem_text = clean_latex(problem_details["problem"])
    answer = clean_latex(problem_details["answer"])
    
    # Extract responses
    responses = problem_details.get("responses", {})
    response_texts = [clean_latex(resp["content"]) for resp in responses.values()]
    response_text = "\n\n".join(response_texts) if response_texts else "No responses available."
    
    # Prepare text for visualization
    formatted_text = (
        f"Problem:\n{problem_text}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Responses:\n{response_text}"
    )
    
    # Display the problem, answer, and responses using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(0.5, 0.5, formatted_text, fontsize=12, ha='center', va='center', wrap=True)
    
    # Save the figure
    output_filename = f"problem_{problem_id}.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Figure saved as {output_filename}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize a JSON problem entry by ID.")
    parser.add_argument("json_path", type=str, help="Path to the JSON file.")
    parser.add_argument("problem_id", type=int, help="ID of the problem to visualize.")
    args = parser.parse_args()
    
    visualize_json(args.json_path, args.problem_id)

if __name__ == "__main__":
    main()