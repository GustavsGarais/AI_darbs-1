import os
from typing import Optional
from app import TextProcessingApp

class ConsoleUI:
    """Console-based user interface for the text processing application"""
    
    def __init__(self):
        """Initialize UI"""
        self.app = TextProcessingApp()
    
    def clear_screen(self):
        """Clear console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print application header"""
        print("\n" + "="*60)
        print(" "*15 + "TEXT PROCESSING APPLICATION")
        print(" "*10 + "Powered by Hugging Face AI Models")
        print("="*60 + "\n")
    
    def print_menu(self):
        """Print main menu"""
        print("\nMAIN MENU")
        print("-"*40)
        print("1. Process new text file")
        print("2. View current results")
        print("3. Export results to file")
        print("4. Exit")
        print("-"*40)
    
    def get_input(self, prompt: str, default: Optional[str] = None) -> str:
        """Get user input with optional default value"""
        if default:
            prompt = f"{prompt} [{default}]: "
        else:
            prompt = f"{prompt}: "
        
        user_input = input(prompt).strip()
        return user_input if user_input else default
    
    def get_int_input(self, prompt: str, default: int, min_val: int = 1, max_val: int = 20) -> int:
        """Get integer input with validation"""
        while True:
            try:
                value = self.get_input(prompt, str(default))
                num = int(value)
                if min_val <= num <= max_val:
                    return num
                else:
                    print(f"Please enter a number between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")
    
    def process_file(self):
        """Handle file processing workflow"""
        print("\n" + "="*60)
        print("PROCESS TEXT FILE")
        print("="*60)
        
        # Get file path
        file_path = self.get_input("\nEnter text file path", "input.txt")
        
        if not os.path.exists(file_path):
            print(f"\n✗ Error: File '{file_path}' not found!")
            input("\nPress Enter to continue...")
            return
        
        # Get parameters
        num_keywords = self.get_int_input(
            "\nNumber of keywords to extract",
            default=5,
            min_val=1,
            max_val=15
        )
        
        num_questions = self.get_int_input(
            "Number of questions to generate",
            default=5,
            min_val=1,
            max_val=10
        )
        
        # Process
        try:
            print("\n⏳ Processing... This may take a minute...\n")
            self.app.process_text(file_path, num_keywords, num_questions)
            
            # Show results
            self.display_results()
            
            # Ask to export
            export = self.get_input("\nExport results to file? (y/n)", "y").lower()
            if export == 'y':
                output_file = self.get_input("Output filename", "results.txt")
                self.app.export_results(output_file)
            
        except Exception as e:
            print(f"\n✗ Error processing file: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def display_results(self):
        """Display current results"""
        content = self.app.get_results()
        
        if not content:
            print("\n✗ No results available. Process a file first.")
            input("\nPress Enter to continue...")
            return
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        print(f"\n📄 File: {content.original_text.file_path}")
        print(f"📊 Word Count: {content.original_text.word_count}")
        
        print("\n" + "-"*60)
        print("📝 SUMMARY")
        print("-"*60)
        # Wrap summary text to 60 characters per line
        summary_words = content.summary.split()
        line = ""
        for word in summary_words:
            if len(line) + len(word) + 1 <= 60:
                line += word + " "
            else:
                print(line.strip())
                line = word + " "
        if line:
            print(line.strip())
        
        print("\n" + "-"*60)
        print("🔑 KEYWORDS")
        print("-"*60)
        for i, kw in enumerate(content.keywords, 1):
            print(f"  {i}. {kw.word}")
        
        print("\n" + "-"*60)
        print("❓ TEST QUESTIONS")
        print("-"*60)
        
        for i, q in enumerate(content.get_valid_questions(), 1):
            print(f"\n{i}. {q.question_text}")
            for j, option in enumerate(q.options):
                marker = "✓" if j == q.correct_answer else " "
                print(f"   {marker} {chr(65+j)}) {option}")
            print(f"   Correct: {chr(65+q.correct_answer)}")
        
        print("\n" + "="*60)
        input("\nPress Enter to continue...")
    
    def export_results_menu(self):
        """Handle results export"""
        content = self.app.get_results()
        
        if not content:
            print("\n✗ No results available. Process a file first.")
            input("\nPress Enter to continue...")
            return
        
        print("\n" + "="*60)
        print("EXPORT RESULTS")
        print("="*60)
        
        output_file = self.get_input("\nOutput filename", "results.txt")
        self.app.export_results(output_file)
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main application loop"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_menu()
            
            choice = self.get_input("\nSelect option (1-4)", "1")
            
            if choice == "1":
                self.process_file()
            elif choice == "2":
                self.display_results()
            elif choice == "3":
                self.export_results_menu()
            elif choice == "4":
                print("\n👋 Thank you for using Text Processing Application!")
                break
            else:
                print("\n✗ Invalid option. Please try again.")
                input("\nPress Enter to continue...")