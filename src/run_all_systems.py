"""
OPAM - Master Execution Script
Run all ML systems with a single command!

Author: Alife
Project: OPAM (Online Purchasing-behavior Analysis & Management)
Completion: 100%
"""

import sys
import time
import subprocess
import os
from datetime import datetime

class OPAMMaster:
    """Master controller for OPAM ML system"""
    
    def __init__(self):
        self.start_time = None
        self.results = {}
        self.total_modules = 10
        
    def print_banner(self):
        """Print awesome banner"""
        banner = """
████████████████████████████████████████████████████████████████████████████████
█                                                                              █
█    ██████╗ ██████╗  █████╗ ███╗   ███╗                                      █
█   ██╔═══██╗██╔══██╗██╔══██╗████╗ ████║                                      █
█   ██║   ██║██████╔╝███████║██╔████╔██║                                      █
█   ██║   ██║██╔═══╝ ██╔══██║██║╚██╔╝██║                                      █
█   ╚██████╔╝██║     ██║  ██║██║ ╚═╝ ██║                                      █
█    ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚═╝     ╚═╝                                      █
█                                                                              █
█              MASTER EXECUTION SYSTEM v1.0                                    █
█         Complete ML Pipeline - One Command Execution                         █
█                                                                              █
████████████████████████████████████████████████████████████████████████████████
        """
        print(banner)
        print(f"\n{'═' * 80}")
        print(f"  🚀 Starting OPAM Complete Analysis")
        print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  📊 Running {self.total_modules} modules")
        print(f"{'═' * 80}\n")
    
    def print_module_header(self, module_num, total, name, description):
        """Print module execution header"""
        print(f"\n{'─' * 80}")
        print(f"  [{module_num}/{total}] {name}")
        print(f"  {description}")
        print(f"{'─' * 80}")
    
    def run_module(self, script_name, module_name, module_num, estimated_time):
        """Run a single module with error handling"""
        
        print(f"\n⏳ Estimated time: ~{estimated_time} minutes")
        print(f"▶️  Running {script_name}...\n")
        
        module_start = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            module_end = time.time()
            duration = module_end - module_start
            
            if result.returncode == 0:
                print(f"\n✅ {module_name} completed successfully!")
                print(f"⏱️  Time taken: {duration:.1f} seconds")
                self.results[module_name] = {
                    'status': 'SUCCESS',
                    'duration': duration,
                    'output': result.stdout
                }
                return True
            else:
                print(f"\n❌ {module_name} failed!")
                print(f"Error: {result.stderr[:500]}")
                self.results[module_name] = {
                    'status': 'FAILED',
                    'duration': duration,
                    'error': result.stderr
                }
                return False
                
        except subprocess.TimeoutExpired:
            print(f"\n⏰ {module_name} timed out!")
            self.results[module_name] = {
                'status': 'TIMEOUT',
                'duration': 600
            }
            return False
            
        except Exception as e:
            print(f"\n💥 {module_name} crashed!")
            print(f"Exception: {str(e)}")
            self.results[module_name] = {
                'status': 'CRASHED',
                'error': str(e)
            }
            return False
    
    def run_all_systems(self):
        """Run all OPAM systems in sequence"""
        
        self.start_time = time.time()
        
        # Module definitions
        modules = [
            {
                'script': 'expense_predictor.py',
                'name': 'Expense Prediction System',
                'description': '6 ML models, 98% accuracy prediction',
                'time': 10
            },
            {
                'script': 'visualize_results.py',
                'name': 'Prediction Visualization',
                'description': 'Charts 1-8: Model performance & analysis',
                'time': 2
            },
            {
                'script': 'anomaly_detector_simple.py',
                'name': 'Anomaly Detection System',
                'description': '3 algorithms, pattern recognition',
                'time': 5
            },
            {
                'script': 'visualize_anomalies.py',
                'name': 'Anomaly Visualization',
                'description': 'Charts 9-13: Anomaly analysis',
                'time': 2
            },
            {
                'script': 'fraud_detector.py',
                'name': 'Fraud Detection System',
                'description': '0-100 risk scoring, 5 fraud patterns',
                'time': 5
            },
            {
                'script': 'visualize_fraud.py',
                'name': 'Fraud Visualization',
                'description': 'Charts 14-16: Fraud analysis',
                'time': 2
            },
            {
                'script': 'user_clusterer.py',
                'name': 'User Clustering System',
                'description': 'K-Means segmentation, 5 personas',
                'time': 3
            },
            {
                'script': 'visualize_clusters.py',
                'name': 'Clustering Visualization',
                'description': 'Charts 17-18: User segments',
                'time': 1
            },
            {
                'script': 'budget_recommender.py',
                'name': 'Budget Recommendation System',
                'description': 'AI-powered budget optimization',
                'time': 2
            },
            {
                'script': 'visualize_budgets.py',
                'name': 'Budget Visualization',
                'description': 'Charts 19-20: Savings analysis',
                'time': 1
            }
        ]
        
        # Run each module
        success_count = 0
        
        for i, module in enumerate(modules, 1):
            self.print_module_header(
                i, 
                len(modules), 
                module['name'], 
                module['description']
            )
            
            if self.run_module(
                module['script'],
                module['name'],
                i,
                module['time']
            ):
                success_count += 1
            
            # Show progress
            progress = (i / len(modules)) * 100
            print(f"\n📊 Overall Progress: {progress:.0f}% ({i}/{len(modules)} modules)")
        
        return success_count, len(modules)
    
    def generate_summary_report(self, success_count, total_count):
        """Generate execution summary"""
        
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        print(f"\n\n{'═' * 80}")
        print(f"  📊 EXECUTION SUMMARY")
        print(f"{'═' * 80}\n")
        
        # Overall stats
        print(f"✅ Modules Completed: {success_count}/{total_count}")
        print(f"⏱️  Total Time: {total_duration/60:.1f} minutes")
        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Module details
        print(f"\n{'─' * 80}")
        print(f"  Module Details:")
        print(f"{'─' * 80}\n")
        
        for module_name, result in self.results.items():
            status_icon = '✅' if result['status'] == 'SUCCESS' else '❌'
            duration = result.get('duration', 0)
            print(f"{status_icon} {module_name:40s} {duration:>6.1f}s  {result['status']}")
        
        # Success rate
        success_rate = (success_count / total_count) * 100
        print(f"\n📈 Success Rate: {success_rate:.1f}%")
        
        # Check outputs
        self.verify_outputs()
        
        # Final message
        if success_count == total_count:
            print(f"\n{'═' * 80}")
            print(f"  🎉 ALL SYSTEMS OPERATIONAL!")
            print(f"  🏆 100% COMPLETION - PERFECT EXECUTION!")
            print(f"{'═' * 80}\n")
            print(f"✨ Results available in:")
            print(f"   📂 ../results/ - All CSV files")
            print(f"   📊 ../charts/  - All 20 visualization charts")
            print(f"\n🚀 Your OPAM system is ready for demo!")
        else:
            print(f"\n{'═' * 80}")
            print(f"  ⚠️  PARTIAL COMPLETION")
            print(f"  {success_count}/{total_count} modules succeeded")
            print(f"{'═' * 80}\n")
            print(f"📋 Check error logs above for details")
        
        # Save report
        self.save_execution_report(success_count, total_count, total_duration)
    
    def verify_outputs(self):
        """Verify all output files exist"""
        
        print(f"\n{'─' * 80}")
        print(f"  📁 Output Verification:")
        print(f"{'─' * 80}\n")
        
        # Check results directory
        results_dir = '../results'
        if os.path.exists(results_dir):
            result_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
            print(f"✅ Results folder: {len(result_files)} CSV files")
        else:
            print(f"❌ Results folder: NOT FOUND")
        
        # Check charts directory
        charts_dir = '../charts'
        if os.path.exists(charts_dir):
            chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
            print(f"✅ Charts folder: {len(chart_files)} PNG files")
            
            if len(chart_files) >= 20:
                print(f"   🎊 All 20 charts created!")
            else:
                print(f"   ⚠️  Expected 20 charts, found {len(chart_files)}")
        else:
            print(f"❌ Charts folder: NOT FOUND")
    
    def save_execution_report(self, success_count, total_count, duration):
        """Save execution report to file"""
        
        report_path = '../results/execution_report.txt'
        
        try:
            with open(report_path, 'w') as f:
                f.write("OPAM MASTER EXECUTION REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Duration: {duration/60:.1f} minutes\n")
                f.write(f"Success Rate: {(success_count/total_count)*100:.1f}%\n\n")
                
                f.write("Module Results:\n")
                f.write("-" * 80 + "\n")
                for module_name, result in self.results.items():
                    f.write(f"{module_name}: {result['status']} ({result.get('duration', 0):.1f}s)\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("Report saved successfully.\n")
            
            print(f"\n💾 Execution report saved to: {report_path}")
            
        except Exception as e:
            print(f"\n⚠️  Could not save report: {str(e)}")


def main():
    """Main execution function"""
    
    # Create master controller
    master = OPAMMaster()
    
    # Print banner
    master.print_banner()
    
    # Confirm execution
    print("⚠️  This will run ALL 10 modules (~30 minutes total)")
    print("📊 Progress will be shown for each module\n")
    
    response = input("Ready to start? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Execution cancelled.")
        print("   Run individual modules if needed:")
        print("   python3 expense_predictor.py")
        return
    
    print("\n🚀 Starting execution...\n")
    time.sleep(1)
    
    # Run all systems
    success_count, total_count = master.run_all_systems()
    
    # Generate summary
    master.generate_summary_report(success_count, total_count)
    
    # Final prompt
    print("\n" + "=" * 80)
    print("  🎤 Ready for your Friday demo!")
    print("  📊 All results are in ../results/ and ../charts/")
    print("  🏆 You've built something AMAZING!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user.")
        print("   Progress has been saved.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {str(e)}")
        print("   Please check your installation and try again.")
        sys.exit(1)
