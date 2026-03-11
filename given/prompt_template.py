
# ============================================================================
# Chainable PromptTemplate Class
# ============================================================================

from typing import List, Dict

class PromptTemplate:
    """
    A chainable prompt template class.
    
    Each method returns a new PromptTemplate instance, allowing fluent
    method chaining to build complex prompts incrementally.
    """
    
    def __init__(self, template: str, **kwargs):
        """
        Initialize the template.
        
        Args:
            template: The base prompt text (can include {placeholders})
            **kwargs: Default values for placeholders
        """
        self.template = template
        self.default_params = kwargs
    
    
    def with_role(self, role: str) -> 'PromptTemplate':
        """
        Add a role/persona to the prompt.
        
        Args:
            role: Description of the role (e.g., "a physics professor")
            
        Returns:
            A new PromptTemplate with the role prepended
        """
        new_template = f"You are {role}.\n\n{self.template}"
        return PromptTemplate(new_template, **self.default_params)
    
    def with_output_format(self, format_spec: str) -> 'PromptTemplate':
        """
        Specify the desired output format.
        
        Args:
            format_spec: Description of expected output format
            
        Returns:
            A new PromptTemplate with format instructions appended
        """
        new_template = f"{self.template}\n\nProvide your response in the following format:\n{format_spec}"
        return PromptTemplate(new_template, **self.default_params)
    
    def with_context(self, context: str) -> 'PromptTemplate':
        """
        Add contextual information to the prompt.
        
        Args:
            context: Background information or context
            
        Returns:
            A new PromptTemplate with context added
        """
        new_template = f"Context: {context}\n\n{self.template}"
        return PromptTemplate(new_template, **self.default_params)
    
    def with_constraints(self, constraints: List[str]) -> 'PromptTemplate':
        """
        Add constraints to the prompt.
        
        Args:
            constraints: List of constraints to follow
            
        Returns:
            A new PromptTemplate with constraints appended
        """
        constraints_text = "\n".join([f"- {c}" for c in constraints])
        new_template = f"{self.template}\n\nConstraints:\n{constraints_text}"
        return PromptTemplate(new_template, **self.default_params)
    
    def with_examples(self, examples: List[Dict[str, str]]) -> 'PromptTemplate':
        """
        Add few-shot examples to the prompt.
        
        Args:
            examples: List of dicts with 'input' and 'output' keys
            
        Returns:
            A new PromptTemplate with examples inserted
        """
        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(examples)
        ])
        new_template = f"Examples:\n{examples_text}\n\n{self.template}"
        return PromptTemplate(new_template, **self.default_params)
    
    def with_chain_of_thought(self) -> 'PromptTemplate':
        """
        Add chain-of-thought instruction.
        
        Returns:
            A new PromptTemplate with step-by-step reasoning instruction
        """
        new_template = f"{self.template}\n\nThink through this step-by-step and show your reasoning."
        return PromptTemplate(new_template, **self.default_params)
    

    def full_prompt(self, **kwargs) -> str:
        """
        Format the template with provided parameters.
        
        Args:
            **kwargs: Values to fill in placeholders
            
        Returns:
            The formatted prompt string
        """
        params = {**self.default_params, **kwargs}
        return self.template.format(**params)
    

    def generate(self, client, model, **kwargs) -> str:
        """
        Use Google genai client to generate response from an input prompt
        """
        full_prompt = self.full_prompt(**kwargs)
        response = client.models.generate_content(model = model, contents = full_prompt)
        return response.text
    
    def __str__(self) -> str:
        """Return the current template string."""
        return self.template
    
    def __repr__(self) -> str:
        """Return a detailed representation."""
        return f"PromptTemplate(template={self.template[:50]}..., params={self.default_params})"


class FeynmanTemplate(PromptTemplate):
    """
    Extended PromptTemplate implementing the Feynman Learning Technique.
    
    The Feynman Technique encourages simple explanations, identifies knowledge
    gaps, and refines understanding through iterative improvement. Inherits all
    chainable methods from PromptTemplate while adding Feynman-specific features.
    """

    def with_feynman_learning(self, target_audience: str = "a child", num_rounds: int = 1) -> 'FeynmanTemplate':
        """
        Apply the Feynman Learning Technique to the prompt.
        
        Structures the prompt to encourage explaining concepts simply, identifying
        knowledge gaps, and refining understanding through iterative cycles.
        
        Args:
            target_audience (str): Who you're explaining to.
                                  Default: "a child"
                                  Examples: "a high school student", "a beginner"
            num_rounds (int): Number of refinement cycles (must be >= 1).
                            Default: 1. Increase for deeper iteration.
            
        Returns:
            FeynmanTemplate: New instance with Feynman structure applied,
                           allowing method chaining with other techniques.
        
        Raises:
            ValueError: If target_audience is empty, None, or not a string
            TypeError: If num_rounds is not a positive integer
            
        Example:
            >>> prompt = FeynmanTemplate("Explain Machine Learning")
            >>> result = prompt.with_feynman_learning("a 10-year-old", num_rounds=2)
            >>> print(result.full_prompt())
        """
        # Input validation
        if not target_audience or not isinstance(target_audience, str):
            raise ValueError("target_audience must be a non-empty string")
        
        if not isinstance(num_rounds, int) or num_rounds < 1:
            raise TypeError("num_rounds must be a positive integer (>= 1)")
        
        # Build the Feynman-structured prompt
        feynman_prompt = f"""{self.template}

Apply the Feynman Learning Technique:

Step 1 - Simple Explanation:
Explain this concept as if teaching {target_audience}. Use simple language, avoid jargon.

Step 2 - Identify Gaps:
Where did you struggle to explain it simply? What parts were unclear?

Step 3 - Refine Your Explanation:
Address the gaps you identified. Improve clarity and simplicity."""
        
        # Add iteration instruction if multiple rounds requested
        if num_rounds > 1:
            feynman_prompt += f"\n\nStep 4 - Iterate and Refine:\nRepeat steps 2-3 for {num_rounds} refinement round(s) to deepen understanding."
        
        # Return as FeynmanTemplate to maintain chainability
        result = FeynmanTemplate(feynman_prompt, **self.default_params)
        return result
    

    
    def with_knowledge_gap_focus(self, specific_gaps: List[str] = None) -> 'FeynmanTemplate':
        """
        Focus Feynman Learning on specific knowledge gaps.
        
        Directs the model to pay special attention to particular areas where
        understanding might be incomplete. Works well with with_feynman_learning().
        
        Args:
            specific_gaps (List[str]): List of specific areas to investigate.
                                      If None, general gaps are identified.
                                      Examples: ["Why does this happen?", "When is this used?"]
            
        Returns:
            FeynmanTemplate: New instance with gap-focused instructions.
        
        Raises:
            TypeError: If specific_gaps is provided but not a list or None
            ValueError: If specific_gaps is an empty list
            
        Example:
            >>> gaps = ["Why mechanism X works", "Real-world applications"]
            >>> prompt = FeynmanTemplate("Explain Photosynthesis")
            >>> result = prompt.with_knowledge_gap_focus(gaps)
        """
        # Input validation
        if specific_gaps is not None:
            if not isinstance(specific_gaps, list):
                raise TypeError("specific_gaps must be a list or None")
            if isinstance(specific_gaps, list) and len(specific_gaps) == 0:
                raise ValueError("specific_gaps list cannot be empty")
        
        # Create gap-focused instruction text
        if not specific_gaps:
            gap_text = "Identify and explain the most important gaps in your understanding."
        else:
            gap_text = "Pay special attention to these knowledge gaps:\n" + "\n".join(
                [f"  - {gap}" for gap in specific_gaps]
            )
        
        new_template = f"""{self.template}

{gap_text}

For each gap, provide a clear, simple explanation that a beginner could understand."""
        
        result = FeynmanTemplate(new_template, **self.default_params)
        return result