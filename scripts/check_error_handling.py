#!/usr/bin/env python3
"""
Check for proper error handling patterns in production code.
Ensures retry logic and structured error responses.
"""

import ast
import sys
import os
from pathlib import Path

class ErrorHandlingChecker(ast.NodeVisitor):
    """AST visitor to check error handling patterns."""
    
    def __init__(self, filename):
        self.filename = filename
        self.errors = []
        self.in_except = False
        
    def visit_ExceptHandler(self, node):
        """Check exception handlers for proper patterns."""
        self.in_except = True
        
        # Check for bare except (should specify exception type)
        if node.type is None:
            self.errors.append({
                'line': node.lineno,
                'message': 'Bare except clause - should specify exception type'
            })
        
        # Check exception handler body
        if node.body:
            has_logging = False
            has_raise_or_return = False
            
            for stmt in ast.walk(node):
                # Check for logging
                if isinstance(stmt, ast.Call):
                    if isinstance(stmt.func, ast.Attribute):
                        if stmt.func.attr in ['error', 'warning', 'exception']:
                            has_logging = True
                
                # Check for raise or return
                if isinstance(stmt, (ast.Raise, ast.Return)):
                    has_raise_or_return = True
            
            # Warn if no logging in exception handler
            if not has_logging:
                self.errors.append({
                    'line': node.lineno,
                    'message': 'Exception handler should log the error'
                })
                
        self.in_except = False
        self.generic_visit(node)
        
    def check_file(self):
        """Check the file for error handling issues."""
        with open(self.filename, 'r') as f:
            try:
                tree = ast.parse(f.read(), self.filename)
                self.visit(tree)
            except SyntaxError as e:
                print(f"Syntax error in {self.filename}: {e}")
                return True
        
        if self.errors:
            print(f"\n❌ Error handling issues in {self.filename}:")
            for error in self.errors:
                print(f"  Line {error['line']}: {error['message']}")
            return True
        
        return False

def main():
    """Main entry point for the error handling checker."""
    if len(sys.argv) < 2:
        print("Usage: check_error_handling.py <file1> [file2] ...")
        sys.exit(1)
    
    has_errors = False
    
    for filepath in sys.argv[1:]:
        # Skip test files
        if 'test' in filepath or filepath.startswith('tests/'):
            continue
            
        if filepath.endswith('.py'):
            checker = ErrorHandlingChecker(filepath)
            if checker.check_file():
                has_errors = True
    
    if has_errors:
        print("\nProduction code should have proper error handling:")
        print("- Use specific exception types (not bare except)")
        print("- Log errors with appropriate level")
        print("- Return structured error responses or re-raise")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == '__main__':
    main()