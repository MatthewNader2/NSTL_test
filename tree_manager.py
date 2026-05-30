# tree_manager.py
import json
import os
import re
import sys

TREES_DIR = "trees"
TEMP_FILE = "temp.json"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def get_tree_files():
    """Returns a list of all JSON files in the trees directory, excluding temp.json."""
    if not os.path.exists(TREES_DIR):
        os.makedirs(TREES_DIR)

    files = [f for f in os.listdir(TREES_DIR) if f.endswith(".json") and f != TEMP_FILE]
    return sorted(files)


def list_trees():
    clear_screen()
    print("=" * 60)
    print(" AVAILABLE DOMAIN TREES")
    print("=" * 60)
    files = get_tree_files()
    if not files:
        print(" No tree files found.")
    else:
        for i, f in enumerate(files):
            try:
                with open(os.path.join(TREES_DIR, f), "r", encoding="utf-8") as file:
                    data = json.load(file)
                    domain = data.get("domain_name", "Unknown Domain")
                    cell_count = len(data.get("cells", []))
                    print(
                        f" [{i + 1}] {f} \n     -> Domain: {domain} | Cells: {cell_count}\n"
                    )
            except Exception as e:
                print(f" [{i + 1}] {f} (ERROR READING: {str(e)})")

    input("\nPress Enter to return to menu...")


def inspect_tree():
    clear_screen()
    files = get_tree_files()
    if not files:
        print("No trees available to inspect.")
        input("Press Enter to return...")
        return

    print("Select a tree to inspect:\n")
    for i, f in enumerate(files):
        print(f" [{i + 1}] {f}")

    try:
        choice = int(input("\nEnter tree number (or 0 to cancel): "))
        if choice == 0:
            return
        selected_file = files[choice - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        input("Press Enter to return...")
        return

    clear_screen()
    print(f"--- CELL IDs IN {selected_file} ---\n")
    try:
        with open(
            os.path.join(TREES_DIR, selected_file), "r", encoding="utf-8"
        ) as file:
            data = json.load(file)
            cells = data.get("cells", [])
            for i, cell in enumerate(cells):
                cell_type = cell.get("type", "micro")
                print(
                    f" {i + 1}. {cell.get('cell_id', 'UNKNOWN')} [{cell_type.upper()}]"
                )
    except Exception as e:
        print(f"Error reading file: {e}")

    input("\nPress Enter to return to menu...")


def merge_temp():
    clear_screen()
    temp_path = os.path.join(TREES_DIR, TEMP_FILE)

    if not os.path.exists(temp_path):
        print(f"'{TEMP_FILE}' does not exist in the '{TREES_DIR}' folder.")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({"cells": []}, f, indent=2)
        print("An empty 'temp.json' has been created for you to paste into.")
        input("\nPress Enter to return...")
        return

    # 1. Safely load and AUTO-CLEAN temp.json
    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # AUTO-CLEANER FOR LLM MISTAKES
        # Remove markdown code blocks
        clean_content = re.sub(r"```json", "", raw_content)
        clean_content = re.sub(r"```", "", clean_content)
        # Remove trailing commas before closing brackets (The #1 LLM JSON error)
        clean_content = re.sub(r",\s*([\]}])", r"\1", clean_content)

        temp_data = json.loads(clean_content)

    except json.JSONDecodeError as e:
        print("=" * 60)
        print(" [FATAL ERROR] INVALID JSON SYNTAX IN temp.json")
        print("=" * 60)
        print(f" The merge was aborted to protect your trees.")
        print(f" Error Details: {str(e)}")
        print(
            "\n TIP: Open temp.json in VS Code and check the line number mentioned above."
        )
        print(" Look for missing commas between cells: }, {")
        input("\nPress Enter to return...")
        return

    # Extract cells smartly
    new_cells = []
    if isinstance(temp_data, list):
        new_cells = temp_data
    elif isinstance(temp_data, dict):
        new_cells = temp_data.get("cells", [])

    if not new_cells:
        print("No valid cells found in temp.json.")
        input("Press Enter to return...")
        return

    # 2. Select target tree
    files = get_tree_files()
    if not files:
        print("No target trees available.")
        input("Press Enter to return...")
        return

    print(f"Found {len(new_cells)} cells in temp.json.\n")
    print("Select the TARGET TREE to merge into:\n")
    for i, f in enumerate(files):
        print(f" [{i + 1}] {f}")

    try:
        choice = int(input("\nEnter tree number (or 0 to cancel): "))
        if choice == 0:
            return
        selected_file = files[choice - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        input("Press Enter to return...")
        return

    # 3. Perform the Merge
    target_path = os.path.join(TREES_DIR, selected_file)
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            target_data = json.load(f)

        target_cells = target_data.get("cells", [])
        existing_ids = {c.get("cell_id") for c in target_cells}

        added_count = 0
        skipped_count = 0

        for new_cell in new_cells:
            c_id = new_cell.get("cell_id")
            if c_id in existing_ids:
                print(f"  [SKIPPED] Cell '{c_id}' already exists in {selected_file}.")
                skipped_count += 1
            else:
                target_cells.append(new_cell)
                existing_ids.add(c_id)
                added_count += 1

        target_data["cells"] = target_cells

        # 4. Save updated tree safely
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(target_data, f, indent=2)

        print(
            f"\n[SUCCESS] Merged {added_count} new cells into {selected_file} ({skipped_count} skipped)."
        )

        # 5. Wipe temp.json
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({"cells": []}, f, indent=2)
        print("[SUCCESS] temp.json has been securely wiped and is ready for new data.")

    except Exception as e:
        print(f"An error occurred during merging: {e}")

    input("\nPress Enter to return to menu...")


def main_menu():
    while True:
        clear_screen()
        print("==========================================================")
        print("  NSTL SECURE TREE MANAGER (Ontology Editor)")
        print("==========================================================")
        print(" [1] List all available domain trees (Overview)")
        print(" [2] Inspect cell IDs of a specific tree")
        print(" [3] Merge 'temp.json' into an existing tree")
        print(" [4] Exit")
        print("==========================================================")

        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            list_trees()
        elif choice == "2":
            inspect_tree()
        elif choice == "3":
            merge_temp()
        elif choice == "4":
            clear_screen()
            print("Exiting Tree Manager. Safe travels in the Lattice!")
            sys.exit(0)
        else:
            print("Invalid selection.")


if __name__ == "__main__":
    main_menu()
