import json
import pandas as pd
from typing import Dict, List, Any, Literal
from datetime import datetime
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict
import os

# =============================================================================
# MODULE 1: Dataset Loading
# =============================================================================

class DatasetLoader:
    """Handles loading and validation of all required JSON datasets."""
    
    def __init__(self):
        self.reference_codes = {}
        self.validation_records = []
        self.test_records = []
        self.insurance_policies = []
    
    def load_all_datasets(self):
        """Load all required datasets from JSON files."""
        try:
            # Load reference codes
            with open('reference_codes.json', 'r') as f:
                self.reference_codes = json.load(f)
            
            # Load validation records
            with open('validation_records.json', 'r') as f:
                self.validation_records = json.load(f)
            
            # Load test records
            with open('test_records.json', 'r') as f:
                self.test_records = json.load(f)
            
            # Load insurance policies
            with open('insurance_policies.json', 'r') as f:
                self.insurance_policies = json.load(f)
                
            print(f"✅ Loaded {len(self.validation_records)} validation records")
            print(f"✅ Loaded {len(self.test_records)} test records")
            print(f"✅ Loaded {len(self.insurance_policies)} insurance policies")
            print(f"✅ Loaded reference codes for CPT and ICD-10")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def get_code_description(self, code: str, code_type: str) -> str:
        """Get human-readable description for medical codes."""
        if code_type.upper() == 'CPT':
            return self.reference_codes.get('CPT', {}).get(code, f"Unknown CPT code: {code}")
        elif code_type.upper() == 'ICD10':
            return self.reference_codes.get('ICD10', {}).get(code, f"Unknown ICD-10 code: {code}")
        return f"Unknown code: {code}"

# Initialize global data loader
data_loader = DatasetLoader()

# =============================================================================
# MODULE 2: Tool Definitions & Implementations
# =============================================================================

@tool
def summarize_patient_record(record_str: str) -> str:
    """
    Summarizes a patient record by extracting and formatting key medical and demographic information.
    
    Args:
        record_str: JSON string containing patient record data
        
    Returns:
        Formatted summary of patient information including demographics, diagnoses, and procedures
    """
    try:
        record = json.loads(record_str)
        
        # Calculate patient age
        dob = datetime.strptime(record['date_of_birth'], '%Y-%m-%d')
        service_date = datetime.strptime(record['date_of_service'], '%Y-%m-%d')
        age = (service_date - dob).days // 365
        
        # Get descriptions for medical codes
        diagnosis_descriptions = []
        for code in record['diagnosis_codes']:
            desc = data_loader.get_code_description(code, 'ICD10')
            diagnosis_descriptions.append(f"{code}: {desc}")
        
        procedure_descriptions = []
        for code in record['procedure_codes']:
            desc = data_loader.get_code_description(code, 'CPT')
            procedure_descriptions.append(f"{code}: {desc}")
        
        summary = f"""
PATIENT SUMMARY:
Patient ID: {record['patient_id']}
Name: {record['name']}
Age: {age} years old
Gender: {record['gender']}
Insurance Policy: {record['insurance_policy_id']}

MEDICAL INFORMATION:
Diagnoses: {'; '.join(diagnosis_descriptions)}
Procedures: {'; '.join(procedure_descriptions)}
Provider: {record['provider_id']} ({record['provider_specialty']})
Location: {record['location']}
Service Date: {record['date_of_service']}
Billed Amount: ${record['billed_amount']:,.2f}

AUTHORIZATION STATUS:
Preauthorization Required: {record['preauthorization_required']}
Preauthorization Obtained: {record['preauthorization_obtained']}
        """.strip()
        
        return summary
        
    except Exception as e:
        return f"Error processing patient record: {str(e)}"

@tool
def summarize_policy_guideline(policy_id: str) -> str:
    """
    Retrieves and summarizes insurance policy guidelines for a specific policy ID.
    
    Args:
        policy_id: The insurance policy identifier
        
    Returns:
        Formatted summary of policy coverage rules and requirements
    """
    try:
        # Find the policy in loaded data
        policy = None
        for p in data_loader.insurance_policies:
            if p['policy_id'] == policy_id:
                policy = p
                break
        
        if not policy:
            return f"Policy {policy_id} not found in database."
        
        summary = f"""
POLICY SUMMARY:
Policy ID: {policy['policy_id']}
Plan Name: {policy['plan_name']}

COVERAGE DETAILS:
"""
        
        for procedure in policy['covered_procedures']:
            proc_desc = data_loader.get_code_description(procedure['procedure_code'], 'CPT')
            
            diagnosis_descriptions = []
            for diag_code in procedure['covered_diagnoses']:
                diag_desc = data_loader.get_code_description(diag_code, 'ICD10')
                diagnosis_descriptions.append(f"{diag_code}: {diag_desc}")
            
            summary += f"""
• Procedure: {procedure['procedure_code']} - {proc_desc}
  - Covered Diagnoses: {'; '.join(diagnosis_descriptions)}
  - Age Range: {procedure['age_range'][0]}-{procedure['age_range'][1]} years
  - Gender: {procedure['gender']}
  - Preauthorization Required: {procedure['requires_preauthorization']}
  - Notes: {procedure['notes']}
"""
        
        return summary.strip()
        
    except Exception as e:
        return f"Error processing policy guidelines: {str(e)}"

@tool
def check_claim_coverage(record_summary: str, policy_summary: str) -> str:
    """
    Analyzes patient record against policy guidelines to determine claim coverage decision.
    
    Args:
        record_summary: Formatted patient record summary
        policy_summary: Formatted policy guidelines summary
        
    Returns:
        Coverage decision with detailed reasoning
    """
    try:
        # Extract key information from summaries using string parsing
        # This is a simplified approach - in production, you'd want more robust parsing
        
        decision_prompt = f"""
Based on the patient record and policy guidelines provided, determine if this medical claim should be APPROVED or DENIED.

PATIENT RECORD:
{record_summary}

POLICY GUIDELINES:
{policy_summary}

Analyze the following criteria:
1. Is the procedure code covered under this policy?
2. Is the diagnosis code eligible for this procedure?
3. Does the patient's age fall within the covered age range?
4. Does the patient's gender match the policy requirements?
5. Are preauthorization requirements met?

Provide your decision as either "APPROVED" or "DENIED" followed by a detailed explanation of your reasoning.

Format your response as:
Decision: [APPROVED/DENIED]
Reason: [Detailed explanation of why the claim was approved or denied, referencing specific policy criteria]
"""
        
        # For this implementation, we'll return a structured analysis
        # In a real system, this would involve an LLM call
        return decision_prompt
        
    except Exception as e:
        return f"Error checking claim coverage: {str(e)}"

# =============================================================================
# MODULE 3: System Instruction Prompt
# =============================================================================

SYSTEM_INSTRUCTION = """
You are an insurance claim processing agent responsible for determining coverage decisions for medical claims.

AVAILABLE TOOLS:
1. summarize_patient_record(record_str): Extracts and formats key patient information from a JSON record
2. summarize_policy_guideline(policy_id): Retrieves and formats insurance policy coverage rules
3. check_claim_coverage(record_summary, policy_summary): Analyzes patient record against policy to make coverage decision

WORKFLOW SEQUENCE:
1. First, use summarize_patient_record() to process the patient's medical record
2. Next, use summarize_policy_guideline() with the patient's policy ID to get coverage rules
3. Finally, use check_claim_coverage() to analyze the patient record against policy guidelines

DECISION CRITERIA:
Evaluate each claim based on:
- Procedure code coverage under the policy
- Diagnosis code eligibility for the procedure
- Patient age within covered age range
- Gender requirements compliance
- Preauthorization requirements satisfaction

OUTPUT FORMAT:
Always provide your final response in exactly this format:

Decision: [APPROVED/DENIED]
Reason: [Detailed explanation referencing specific policy criteria, patient demographics, and medical codes. Include the specific procedure and diagnosis descriptions in plain language.]

IMPORTANT GUIDELINES:
- Be thorough in your analysis of all coverage criteria
- Reference specific medical code descriptions (not just code numbers)
- Clearly state which criteria are met or not met
- For denials, specify exactly which requirement(s) failed
- For approvals, confirm all requirements are satisfied
- Always maintain a professional, clear tone in your reasoning
"""

# =============================================================================
# MODULE 4: LLM Integration
# =============================================================================

class LLMConfig:
    """Configuration for LLM integration using Ollama."""
    
    def __init__(self):
        self.model_name = "qwen2.5:7b"  # Using Ollama qwen2.5:7b model
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.temperature = 0.1  # Low temperature for consistent, factual responses
        
    def get_llm(self):
        """Initialize and return the Ollama LLM client."""
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout=300,  # 5 minute timeout for longer responses
            num_predict=2048  # Max tokens for response generation
        )

# =============================================================================
# MODULE 5: Agent Creation
# =============================================================================

class InsuranceClaimAgent:
    """ReAct-style agent for processing insurance claims using LangGraph."""
    
    def __init__(self):
        self.llm_config = LLMConfig()
        self.llm = self.llm_config.get_llm()
        self.tools = [summarize_patient_record, summarize_policy_guideline, check_claim_coverage]
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create ReAct agent using LangGraph's built-in method."""
        return create_react_agent(
            model=self.llm,
            tools=self.tools,
        )
    
    def process_claim(self, patient_record: Dict[str, Any]) -> str:
        """
        Process a single insurance claim through the agent.
        
        Args:
            patient_record: Dictionary containing patient record data
            
        Returns:
            Agent's coverage decision and reasoning
        """
        try:
            # Step 1: Summarize patient record
            record_summary = summarize_patient_record(json.dumps(patient_record))

            # Step 2: Summarize policy guidelines
            policy_summary = summarize_policy_guideline(patient_record["insurance_policy_id"])

            # Step 3: Build custom prompt for decision
            prompt = f"""
		You are an insurance claim processor. Based on the following patient record and policy:

		PATIENT SUMMARY:
		{record_summary}

		POLICY SUMMARY:
		{policy_summary}

		Please determine if the claim should be approved or denied.
		Respond in ONE SINGLE SENTENCE, in this exact format:

		"The claim for [procedure description(s)] (CPT code(s) …) with diagnosis [ICD code(s)] was [APPROVED/DENIED] because [reason]."
		"""

            result = self.llm.invoke([HumanMessage(content=prompt)])
            response = result.content if hasattr(result, "content") else str(result)

            # Clean for CSV (remove line breaks, extra spaces)
            clean_response = " ".join(response.replace("\n", " ").split())

            return clean_response

        except Exception as e:
            return f"Error processing claim: {str(e)}"

# =============================================================================
# MODULE 6: Agent Experimentation & Validation
# =============================================================================

class AgentValidator:
    """Handles validation and evaluation of the agent's performance."""
    
    def __init__(self, agent: InsuranceClaimAgent):
        self.agent = agent
        self.validation_results = []
    
    def validate_on_dataset(self, validation_records: List[Dict[str, Any]]):
        """Run agent on validation dataset and collect results."""
        print("🔄 Running validation on dataset...")
        
        for i, record in enumerate(validation_records):
            print(f"Processing validation record {i+1}/{len(validation_records)}: {record['patient_id']}")
            
            try:
                response = self.agent.process_claim(record)
                
                result = {
                    'patient_id': record['patient_id'],
                    'agent_response': response,
                    'original_record': record
                }
                
                self.validation_results.append(result)
                
            except Exception as e:
                print(f"❌ Error processing {record['patient_id']}: {str(e)}")
                
                result = {
                    'patient_id': record['patient_id'],
                    'agent_response': f"Error: {str(e)}",
                    'original_record': record
                }
                
                self.validation_results.append(result)
        
        print(f"✅ Validation complete. Processed {len(self.validation_results)} records.")
    
    def analyze_results(self):
        """Analyze validation results for patterns and potential issues."""
        print("\n📊 VALIDATION ANALYSIS:")
        print(f"Total records processed: {len(self.validation_results)}")
        
        # Count decisions
        approved_count = 0
        denied_count = 0
        error_count = 0
        
        for result in self.validation_results:
            response = result['agent_response'].lower()
            if 'approved' in response and 'decision:' in response:
                approved_count += 1
            elif 'denied' in response and 'decision:' in response:
                denied_count += 1
            else:
                error_count += 1
        
        print(f"Approved: {approved_count}")
        print(f"Denied: {denied_count}")
        print(f"Errors/Unclear: {error_count}")
        
        # Show sample responses
        print("\n📝 SAMPLE RESPONSES:")
        for i, result in enumerate(self.validation_results[:3]):
            print(f"\n--- Sample {i+1}: {result['patient_id']} ---")
            print(result['agent_response'][:300] + "..." if len(result['agent_response']) > 300 else result['agent_response'])

# =============================================================================
# MODULE 7: Testing & Final Output Generation
# =============================================================================

class TestRunner:
    """Handles testing on the final dataset and generates submission file."""
    
    def __init__(self, agent: InsuranceClaimAgent):
        self.agent = agent
        self.test_results = []
    
    def run_tests(self, test_records: List[Dict[str, Any]]):
        """Run agent on test dataset."""
        print("🧪 Running tests on final dataset...")
        
        for i, record in enumerate(test_records):
            print(f"Processing test record {i+1}/{len(test_records)}: {record['patient_id']}")
            
            try:
                response = self.agent.process_claim(record)
                
                # Clean up response for CSV output
                clean_response = response.replace('\n', ' ').replace('\r', ' ')
                clean_response = ' '.join(clean_response.split())  # Remove extra whitespace
                
                self.test_results.append({
                    'patient_id': record['patient_id'],
                    'reference_response': clean_response
                })
                
            except Exception as e:
                print(f"❌ Error processing {record['patient_id']}: {str(e)}")
                
                self.test_results.append({
                    'patient_id': record['patient_id'],
                    'reference_response': f"Error processing claim: {str(e)}"
                })
    
    def generate_submission_file(self, filename: str = "submission.csv"):
        """Generate final submission CSV file."""
        if not self.test_results:
            print("❌ No test results to save. Run tests first.")
            return
        
        df = pd.DataFrame(self.test_results)
        df.to_csv(filename, index=False)
        print(f"✅ Submission file saved as '{filename}'")
        print(f"📄 Generated {len(self.test_results)} test responses")

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """Main execution pipeline that runs all modules in sequence."""
    
    print("🚀 Insurance Claim Processing Agent - Starting Pipeline")
    print("=" * 60)
    
    try:
        # Module 1: Load datasets
        print("\n📂 MODULE 1: Loading Datasets")
        data_loader.load_all_datasets()
        
        # Module 5: Create agent
        print("\n🤖 MODULE 5: Creating Insurance Claim Agent")
        agent = InsuranceClaimAgent()
        print("✅ Agent created successfully")
        
        # Module 6: Validation
        print("\n🔍 MODULE 6: Agent Validation")
        validator = AgentValidator(agent)
        validator.validate_on_dataset(data_loader.validation_records)
        validator.analyze_results()
        
        # Module 7: Testing and submission
        print("\n🧪 MODULE 7: Final Testing")
        test_runner = TestRunner(agent)
        test_runner.run_tests(data_loader.test_records)
        test_runner.generate_submission_file()
        
        print("\n🎉 Pipeline completed successfully!")
        print("📄 Check 'submission.csv' for final results")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

# Example usage and testing
if __name__ == "__main__":  
    print("🔧 Execution Starts :)")
    
    # Test data loading
    try:
        main()
        print("Process completed successfully")
        
    except Exception as e:
        print(f"❌ Process failed: {str(e)}")
