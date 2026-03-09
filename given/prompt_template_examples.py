"""
Chainable PromptTemplate Usage Examples with Google Gemini
==========================================================

This script demonstrates how to use the chainable/fluent PromptTemplate 
class with Google's Gemini model. This design pattern allows you to 
build prompts incrementally using method chaining.

Setup:
    pip install google-genai

Usage:
    1. Set your API key as an environment variable:
       export GOOGLE_API_KEY="your-api-key"
    
    2. Run this script:
       python chainable_prompt_template_examples.py
"""

import os
from google import genai

from prompt_template import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

model = "gemini-2.5-flash"
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

generate_prompt_only = True

# ============================================================================
# Example 1: Basic Task Only
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 1: Basic Task Only")
print("=" * 70)

# Create a simple template
prompt = PromptTemplate("Explain Quantum Computing")

print("\n[Constructed Prompt]")
print(prompt.full_prompt())

if not generate_prompt_only:
    print("\n[Response]")
    response = prompt.generate(client, model)
    print(response)


# ============================================================================
# Example 2: Adding a Role
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Adding a Role")
print("=" * 70)

prompt = (PromptTemplate("Explain Quantum Computing")
    .with_role("a physics professor who makes complex topics simple"))

print("\n[Constructed Prompt]")
print(prompt.full_prompt())

if not generate_prompt_only:
    print("\n[Response]")
    response = prompt.generate(client, model)
    print(response)



# ============================================================================
# Example 3: Role + Output Format
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Role + Output Format (Method Chaining)")
print("=" * 70)

prompt = (PromptTemplate("Explain Quantum Computing")
    .with_role("a science educator")
    .with_output_format("""1. Simple Definition (1-2 sentences)
2. Key Concepts (bullet points)
3. Real-World Example
4. Why It Matters"""))

print("\n[Constructed Prompt]")
print(prompt.full_prompt())

if not generate_prompt_only:
    print("\n[Response]")
    response = prompt.generate(client, model)
    print(response)


# ============================================================================
# Example 4: Adding Context
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: With Context")
print("=" * 70)

prompt = (PromptTemplate("Explain Quantum Computing")
    .with_role("a patient tutor")
    .with_context("The student is a high school senior interested in computer science but has no physics background."))

print("\n[Constructed Prompt]")
print(prompt.full_prompt())

if not generate_prompt_only:
    print("\n[Response]")
    response = prompt.generate(client, model)
    print(response)


# ============================================================================
# Example 5: Adding Constraints
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: With Constraints")
print("=" * 70)

prompt = (PromptTemplate("Explain Quantum Computing")
    .with_role("a science communicator")
    .with_constraints([
        "Keep the explanation under 150 words",
        "Use at least one analogy from everyday life",
        "Avoid mathematical equations",
        "End with a thought-provoking question"
    ]))

print("\n[Constructed Prompt]")
print(prompt.full_prompt())

if not generate_prompt_only:
    print("\n[Response]")
    response = prompt.generate(client, model)
    print(response)


# ============================================================================
# Example 6: With Few-Shot Examples
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: With Few-Shot Examples")
print("=" * 70)

prompt = (PromptTemplate("Explain Quantum Computing")
    .with_examples([
        {
            "input": "Explain Machine Learning",
            "output": "Imagine teaching a dog tricks by giving treats for correct behavior. Machine Learning works similarly—we 'reward' computer programs when they make correct predictions, helping them learn patterns from data without explicit programming."
        },
        {
            "input": "Explain Blockchain",
            "output": "Think of a shared Google Doc that everyone can view but no one can secretly edit. Each change is recorded permanently with a timestamp. Blockchain is this concept applied to financial transactions and data—transparent, permanent, and trustworthy without needing a central authority."
        }
    ]))

print("\n[Constructed Prompt]")
print(prompt.full_prompt())

if not generate_prompt_only:
    print("\n[Response]")
    response = prompt.generate(client, model)
    print(response)


# ============================================================================
# Example 7: Chain of Thought
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 7: With Chain-of-Thought Reasoning")
print("=" * 70)

prompt = (PromptTemplate("Explain why Quantum Computing is faster than classical computing for certain problems")
    .with_role("a computer scientist")
    .with_chain_of_thought())

print("\n[Constructed Prompt]")
print(prompt.full_prompt())

if not generate_prompt_only:
    print("\n[Response]")
    response = prompt.generate(client, model)
    print(response)


# ============================================================================
# Example 8: Full Chain - All Techniques Combined
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 8: Full Method Chain - All Techniques")
print("=" * 70)

prompt = (PromptTemplate("Explain Quantum Computing")
    .with_role("an award-winning science journalist")
    .with_context("Writing for a tech newsletter read by busy professionals")
    .with_examples([
        {
            "input": "Explain 5G",
            "output": "Your current phone internet is like a two-lane road. 5G is a 100-lane superhighway. This isn't just about faster cat videos—it's the infrastructure for self-driving cars, remote surgery, and smart cities. The one thing to remember: 5G is less about speed and more about enabling technologies that don't exist yet."
        }
    ])
    .with_output_format("""- Hook (1 attention-grabbing sentence)
- Core Idea (2-3 sentences, use an analogy)  
- Why It Matters (1-2 sentences on real-world impact)
- The Takeaway (1 memorable closing sentence)""")
    .with_constraints([
        "Total response: 100-150 words",
        "Conversational, not academic tone",
        "No jargon without immediate explanation"
    ]))

print("\n[Constructed Prompt]")
print(prompt.full_prompt())

if not generate_prompt_only:
    print("\n[Response]")
    response = prompt.generate(client, model)
    print(response)


# ============================================================================
# Example 9: Using Placeholders
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 9: Using Placeholders for Reusable Templates")
print("=" * 70)

# Create a reusable template with placeholders
explanation_template = (PromptTemplate("Explain {topic} to {audience}")
    .with_role("a {expertise} expert")
    .with_constraints([
        "Keep it under {word_limit} words",
        "Use analogies appropriate for the audience"
    ]))

print("[Reusable Template Created]")
print("Template:", explanation_template.template)
print()

# Use the template for Quantum Computing
print("--- Using template for Quantum Computing ---")
prompt1 = explanation_template.full_prompt(
    topic="Quantum Computing",
    audience="a 10-year-old",
    expertise="physics",
    word_limit="100"
)
print("\n[Formatted Prompt]")
print(prompt1)

# Reuse the same template for a different topic
print("\n--- Reusing template for Blockchain ---")
prompt2 = explanation_template.full_prompt(
    topic="Blockchain",
    audience="a business executive",
    expertise="fintech",
    word_limit="150"
)
print("\n[Formatted Prompt]")
print(prompt2)



# ============================================================================
# Example 10: Comparison - No Technique vs Full Template
# ============================================================================

print("\n" + "=" * 70)
print("COMPARISON: No Technique vs Full Template")
print("=" * 70)

if not generate_prompt_only:
    print("\n--- WITHOUT prompt engineering ---")
    simple_response = client.models.generate_content("Explain Quantum Computing")
    print(simple_response.text)

    print("\n--- WITH full prompt engineering chain ---")
    engineered_prompt = (PromptTemplate("Explain Quantum Computing")
                         .with_role("a science communicator known for clarity")
                         .with_context("Reader has 2 minutes and wants practical understanding")
                         .with_output_format("Hook → Core Concept → One Application → Memorable Takeaway")
                         .with_constraints([
                             "Under 120 words",
                             "One concrete analogy",
                             "No equations or jargon"
                         ]))

    engineered_response = engineered_prompt.generate(client, model)
    print(engineered_response)

print("\n" + "=" * 70)
print("KEY INSIGHT: The chainable design lets you build prompts incrementally,")
print("making it easy to add or remove techniques as needed.")
print("=" * 70)
