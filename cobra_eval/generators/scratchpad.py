"""
Two-pass (or multi-pass) scratchpad reasoning generator.
"""
from PIL import Image
from typing import Any, List
from ..interfaces import BaseGenerator, GenerationResult
from ..registry import Registry
from .utils import clean_reasoning_trace

@Registry.register_generator("scratchpad")
class ScratchpadGenerator(BaseGenerator):
    def __init__(self, vlm: Any, scratchpad_passes: int = 1):
        """
        Initialize with VLM and number of reasoning passes.
        """
        self.vlm = vlm
        self.scratchpad_passes = scratchpad_passes
        
        # Default prompts
        self.reasoning_prompt = (
            "Analyze this image systematically. First, identify all major objects and their colors, "
            "sizes, and positions. Then describe the spatial relationships between objects (left, right, "
            "in front of, behind, etc.). Finally, note any distinctive features, actions, or context. "
            "Be specific and detailed. Do NOT write a caption - only provide your analytical observations."
        )

    def generate(self, image: Image.Image, prompt: str, **kwargs) -> GenerationResult:
        reasoning_max_tokens = kwargs.get('reasoning_max_tokens', 256)
        repetition_penalty = kwargs.get('repetition_penalty', 1.2)
        
        # Clean kwargs for VLM calls
        gen_kwargs = {k: v for k, v in kwargs.items() 
                     if k not in ['reasoning_max_tokens', 'repetition_penalty']}
        
        reasoning_kwargs = gen_kwargs.copy()
        reasoning_kwargs['max_new_tokens'] = reasoning_max_tokens
        reasoning_kwargs['repetition_penalty'] = repetition_penalty

        traces: List[str] = []
        current_context = ""
        
        # Multi-pass reasoning
        for i in range(self.scratchpad_passes):
            prompt_builder = self.vlm.get_prompt_builder()
            
            # For subsequent passes, we could potentially append previous reasoning
            # For now, we keep it simple: if passes > 1, we might just be doing iterative refinement 
            # or simply repeating the analysis (which isn't very useful without changing prompt).
            # Let's implement a refinement chain if passes > 1
            
            if i == 0:
                msg = self.reasoning_prompt
            else:
                msg = f"Review your previous analysis:\n{traces[-1]}\n\nRefine and expand on this analysis. Missed anything? Be more specific."
                
            prompt_builder.add_turn(role="human", message=msg)
            prompt_text = prompt_builder.get_prompt()
            
            trace = self.vlm.generate(
                image,
                prompt_text,
                use_cache=True,
                **reasoning_kwargs
            )
            
            cleaned_trace = clean_reasoning_trace(trace)
            traces.append(cleaned_trace)
            
            # Stop if trace is empty
            if not cleaned_trace:
                break

        # Final consolidation of traces
        final_reasoning = "\n\n".join([t for t in traces if t])
        
        # Second pass: Generate final caption
        prompt_builder_caption = self.vlm.get_prompt_builder()
        
        if not final_reasoning or len(final_reasoning.split()) < 5:
            combined_prompt = prompt
        else:
            combined_prompt = (
                f"{prompt}\n\n"
                f"Based on the following detailed analysis:\n{final_reasoning}\n\n"
                f"Now provide a concise, natural caption that summarizes the key elements:"
            )
            
        prompt_builder_caption.add_turn(role="human", message=combined_prompt)
        caption_prompt_text = prompt_builder_caption.get_prompt()
        
        final_caption = self.vlm.generate(
            image,
            caption_prompt_text,
            use_cache=True,
            **gen_kwargs
        )
        
        return GenerationResult(
            caption=final_caption,
            reasoning_trace=final_reasoning,
            metadata={
                "method": "scratchpad",
                "passes": self.scratchpad_passes,
                "individual_traces": traces
            }
        )

