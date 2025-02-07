import json
import argparse
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

def visualize_json(file_path, entry_id):
    console = Console()
    
    # Load JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if entry exists
    if entry_id not in data:
        console.print(Panel(f"[bold red]Error:[/] Task ID {entry_id} not found.", expand=False))
        return
    
    task_info = data[entry_id]
    console.print(Panel(f"[bold]Task ID:[/] {entry_id}\n[bold]Difficulty:[/] {task_info['difficulty']}", expand=False))
    console.print(Panel(f"[bold]Prompt:[/]\n{task_info['prompt']}", title="Prompt", expand=False))
    
    # Process responses
    responses = task_info.get("responses", {})
    for temperature, response in responses.items():
        content = response.get("content", "")
        code_start = content.find("```python\n")
        code_end = content.find("```", code_start + 9)
        
        if code_start != -1 and code_end != -1:
            before_code = content[:code_start].strip()
            code_block = content[code_start+10:code_end].strip()
            after_code = content[code_end+3:].strip()
            
            console.print(Panel(f"[bold]Temperature:[/] {temperature}", expand=False))
            console.print(before_code)
            syntax = Syntax(code_block, "python", theme="monokai", line_numbers=True)
            console.print(syntax)
            console.print(after_code)
        else:
            console.print(Panel(f"[bold]Temperature:[/] {temperature}\n{content}", expand=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a specific entry in a JSON file.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the JSON file.")
    parser.add_argument("--entry_id", type=str, required=True, help="ID of the entry to visualize.")
    args = parser.parse_args()
    
    visualize_json(args.file_path, args.entry_id)

