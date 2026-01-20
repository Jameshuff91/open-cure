#!/usr/bin/env python3
"""Update evaluation_tasks.json with AF research completion."""

import json
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    tasks_file = base_dir / "evaluation_tasks.json"

    # Load tasks
    with open(tasks_file, 'r') as f:
        tasks_data = json.load(f)

    # Find AF task (id=9)
    for task in tasks_data['tasks']:
        if task['id'] == 9 and task['disease'] == 'Atrial fibrillation':
            task['drugs_researched'] = 30
            task['confirmed_found'] = 1
            task['complete'] = True
            task['stop_reason'] = 'max_candidates_reached'

            print(f'✓ Updated Atrial fibrillation task:')
            print(f'  drugs_researched: 30')
            print(f'  confirmed_found: 1')
            print(f'  complete: True')
            print(f'  stop_reason: max_candidates_reached')
            break

    # Save updated tasks
    with open(tasks_file, 'w') as f:
        json.dump(tasks_data, f, indent=2)

    print(f'\n✓ Updated evaluation_tasks.json')

    # Show summary
    complete_count = len([t for t in tasks_data['tasks'] if t['complete']])
    total_count = len(tasks_data['tasks'])

    print(f'\nOverall progress: {complete_count}/{total_count} diseases complete')

if __name__ == "__main__":
    main()
