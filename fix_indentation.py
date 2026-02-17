#!/usr/bin/env python3
"""Fix indentation for run_lqr_with_manipulator_tracking method."""

def fix_indentation():
    # Read the file
    with open('script_cart_pendulam_2d_extended_ofc_v2.py', 'r') as f:
        lines = f.readlines()
    
    # Find the method
    method_start = None
    for i, line in enumerate(lines):
        if 'def run_lqr_with_manipulator_tracking(self' in line:
            method_start = i
            break
    
    if method_start is None:
        print("Method not found!")
        return
    
    print(f"Method starts at line {method_start + 1}")
    
    # Fix lines starting from line 3230 (index 3229) to line 3750
    # These lines need 4 more spaces of indentation (from 4 to 8 spaces)
    start_fix = 3229  # Line 3230 (0-indexed)
    end_fix = 3750    # Approximate end of method
    
    fixed_lines = lines[:start_fix]
    fixes = 0
    
    for i in range(start_fix, min(end_fix, len(lines))):
        line = lines[i]
        # Stop when we hit the next top-level definition
        if line.startswith('def ') or line.startswith('class ') or (line.startswith('#') and '=' * 50 in line):
            print(f"Found end of method at line {i+1}")
            fixed_lines.extend(lines[i:])
            break
        
        # Fix indentation: add 4 spaces if the line has content
        if line.strip():  # Non-empty line
            current_indent = len(line) - len(line.lstrip())
            if current_indent == 4:  # Should be 8
                fixed_lines.append('    ' + line)
                fixes += 1
            elif current_indent == 0:  # Top-level comment, should be 8
                fixed_lines.append('        ' + line)
                fixes += 1
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    print(f"Fixed {fixes} lines")
    
    # Write back
    with open('script_cart_pendulam_2d_extended_ofc_v2.py', 'w') as f:
        f.writelines(fixed_lines)
    
    print("Done!")

if __name__ == '__main__':
    fix_indentation()
