from pathlib import Path

with open("project_code_AE.md", "w", encoding="utf-8") as salida:
    for root in sorted(Path(".").rglob("*.py")):
        if ".venv" in root.parts or "venv" in root.parts:
            continue
        salida.write(f"\n## File: `{root}`\n```python\n")
        salida.write(root.read_text(encoding="utf-8"))
        salida.write("\n```\n")