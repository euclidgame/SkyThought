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

def wrap_text(text, max_words=12):
    """Wraps text to fit within a certain number of words per line while preserving line breaks."""
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(current_line) >= max_words:
                lines.append(" ".join(current_line))
                current_line = []
        
        if current_line:
            lines.append(" ".join(current_line))
        
        lines.append("")  # Preserve paragraph breaks
    
    return "\n".join(lines)

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
    problem_text = wrap_text(clean_latex(problem_details["prompt"]))
    answer = clean_latex(problem_details["answer"])
    
    # Extract responses
    responses = problem_details.get("responses", {})
    response_texts = [clean_latex(resp["content"]) for resp in responses.values()]
    response_text = "\n\n".join(response_texts) if response_texts else "No responses available."
    
    # Display the problem, answer, and responses using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis("off")
    
    y_pos = 1.0  # Start from the top
    line_spacing = 0.035  # Reduce space between lines
    font_size = 14  # Increase font size
    
    def add_text(ax, text, y, fontsize=font_size, color='black', weight='normal'):
        lines = text.split("\n")
        for line in lines:
            y -= line_spacing
            ax.text(0.05, y, line, fontsize=fontsize, color=color, weight=weight, ha='left', va='top')
        return y - line_spacing  # Adjust for next section
    
    y_pos = add_text(ax, "Prompt:", y_pos, fontsize=font_size + 2, color='blue', weight='bold')
    y_pos = add_text(ax, problem_text, y_pos)
    y_pos = add_text(ax, "Answer:", y_pos, fontsize=font_size + 2, color='green', weight='bold')
    y_pos = add_text(ax, answer, y_pos)
    y_pos = add_text(ax, "Responses:", y_pos, fontsize=font_size + 2, color='cyan', weight='bold')
    y_pos = add_text(ax, response_text, y_pos)
    
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