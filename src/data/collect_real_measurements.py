"""
Real Energy Measurement Collector using CodeCarbon.

This module collects real energy measurements by running actual LLM inference
and measuring power consumption with CodeCarbon. The collected data serves as
a validation set to sanity-check the synthetic training data model.

Methodology:
1. Run actual model inference (T5-small, DistilBERT, etc.)
2. Measure energy consumption with CodeCarbon
3. Extract prompt features using NLP parser
4. Save measurements for model validation

Usage:
    python src/data/collect_real_measurements.py --num_samples 100
    
    # Or programmatically:
    from src.data.collect_real_measurements import collect_measurements
    df = collect_measurements(num_samples=100)

References:
- CodeCarbon: https://codecarbon.io/
- "No public per-prompt energy dataset exists, which is consistent with 
   recent literature highlighting opacity in AI energy reporting."
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

import numpy as np
import pandas as pd

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings during measurement
warnings.filterwarnings('ignore')

# Check for CodeCarbon availability
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("WARNING: CodeCarbon not installed. Install with: pip install codecarbon")

# Try eco2ai as alternative
try:
    import eco2ai
    ECO2AI_AVAILABLE = True
except ImportError:
    ECO2AI_AVAILABLE = False


@dataclass
class RealMeasurement:
    """Single real energy measurement."""
    measurement_id: str
    prompt: str
    category: str
    
    # Features (from NLP parser)
    token_count: int
    word_count: int
    char_count: int
    complexity_score: float
    avg_word_length: float
    avg_sentence_length: float
    
    # Model info
    model_name: str
    model_params_millions: float
    
    # Real measurements
    energy_kwh: float
    energy_joules: float
    duration_seconds: float
    carbon_kg: float
    
    # Metadata
    cpu_model: str
    gpu_available: bool
    timestamp: str


class RealMeasurementCollector:
    """
    Collects real energy measurements using CodeCarbon.
    
    Runs actual model inference and measures energy consumption
    to create a validation dataset for the synthetic model.
    """
    
    # Test prompts across different categories and lengths
    # IMPORTANT: Include prompts up to 200+ tokens for proper scaling validation
    TEST_PROMPTS = {
        "simple": [
            "What is machine learning?",
            "Define artificial intelligence.",
            "Explain Python briefly.",
            "What is a neural network?",
            "What does API mean?",
        ],
        "medium": [
            "How can I improve my Python programming skills effectively and become a better developer?",
            "What are the best practices for writing clean, maintainable code in modern software development?",
            "How do I get started with machine learning projects and what tools should I learn first?",
            "Explain the differences between supervised learning, unsupervised learning, and reinforcement learning in machine learning.",
            "What are the key principles of object-oriented programming and how do they apply to Python development?",
        ],
        "long": [
            "Explain how neural networks learn patterns from data through the process of backpropagation, including forward pass, loss calculation, gradient computation, and weight updates. Also describe how different activation functions like ReLU, sigmoid, and tanh affect the learning process.",
            "Describe the complete process of training a machine learning model from scratch, including data collection, preprocessing, feature engineering, model selection, hyperparameter tuning, cross-validation, training, evaluation, and deployment to production.",
            "Walk me through the steps of building a REST API in Python using Flask or FastAPI, including setting up routes, handling requests and responses, implementing authentication, connecting to a database, and deploying the application.",
            "Provide a comprehensive explanation of the transformer architecture used in modern language models, including self-attention mechanisms, multi-head attention, positional encoding, feed-forward layers, and how these components work together.",
            "Break down the concept of transfer learning in deep learning, explaining how pre-trained models can be fine-tuned for specific tasks, what layers to freeze or unfreeze, and best practices for achieving good results.",
        ],
        "very_long": [
            "Provide a comprehensive analysis of the transformer architecture including self-attention mechanisms, positional encoding, multi-head attention, layer normalization, residual connections, and the encoder-decoder structure. Explain how these components work together for natural language processing tasks like translation, summarization, and question answering. Also discuss the computational complexity and memory requirements of transformers compared to RNNs and LSTMs.",
            "Develop a detailed strategy for building a production-ready machine learning pipeline considering all stages from data ingestion and validation, through feature engineering and selection, model training with cross-validation, hyperparameter optimization using grid search or Bayesian methods, model evaluation with appropriate metrics, deployment using containerization and orchestration, monitoring for data drift and model degradation, and establishing feedback loops for continuous improvement.",
            "Create an exhaustive comparison of different deep learning frameworks including TensorFlow, PyTorch, JAX, and MXNet across multiple dimensions such as ease of use and learning curve, computational performance and optimization capabilities, distributed training support, mobile and edge deployment options, community support and ecosystem, debugging and profiling tools, and production deployment considerations. Provide specific examples and use cases where each framework excels.",
            "Analyze the ethical, social, and economic implications of large language models in society, including concerns about bias and fairness, misinformation and deepfakes, job displacement and economic disruption, privacy and data security, environmental impact from training, intellectual property and copyright issues, and propose comprehensive governance frameworks for responsible AI development and deployment.",
            "Design a complete system architecture for a real-time recommendation engine that handles millions of users, including components for data collection and event streaming, feature store with real-time and batch features, model training infrastructure, model serving with low latency requirements, A/B testing framework for experimentation, feedback loops for continuous learning, monitoring and alerting systems, and considerations for scalability and fault tolerance.",
        ],
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the collector.
        
        Args:
            output_dir: Directory to save measurements
        """
        self.output_dir = output_dir or Path(__file__).parent.parent.parent / "data" / "validation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.measurements: List[RealMeasurement] = []
        self.model = None
        self.tokenizer = None
        self.model_name = None
        
    def _load_model(self, model_name: str = "t5-small"):
        """Load the model for inference."""
        try:
            import torch
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            self.model_name = model_name
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            # Get model size
            num_params = sum(p.numel() for p in self.model.parameters())
            self.model_params_millions = num_params / 1e6
            
            print(f"Loaded {model_name} ({self.model_params_millions:.1f}M parameters)")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def _extract_features(self, prompt: str) -> Dict:
        """Extract NLP features from prompt."""
        try:
            from nlp.parser import parse_prompt
            from nlp.complexity_score import compute_complexity
            
            parsed = parse_prompt(prompt, use_embeddings=False)
            complexity = compute_complexity(prompt)
            
            return {
                "token_count": parsed.token_count,
                "word_count": parsed.word_count,
                "char_count": parsed.char_count,
                "complexity_score": complexity,
                "avg_word_length": parsed.avg_word_length,
                "avg_sentence_length": parsed.avg_sentence_length,
            }
        except Exception as e:
            # Fallback to simple calculation
            words = prompt.split()
            return {
                "token_count": len(words) + 2,  # Approximate
                "word_count": len(words),
                "char_count": len(prompt),
                "complexity_score": 0.5,
                "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
                "avg_sentence_length": len(words),
            }
    
    def _run_inference(self, prompt: str) -> Tuple[str, float]:
        """
        Run model inference and return output and duration.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Tuple of (output_text, duration_seconds)
        """
        import torch
        
        # Prepare input
        input_text = f"summarize: {prompt}"
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        # Run inference
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=100,
                min_length=10,
                num_beams=4,
                early_stopping=True,
            )
        
        duration = time.perf_counter() - start_time
        
        # Decode output
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return output_text, duration
    
    def _get_cpu_info(self) -> str:
        """Get CPU model information."""
        try:
            import platform
            return platform.processor() or "Unknown CPU"
        except:
            return "Unknown CPU"
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def measure_single(self, prompt: str, category: str, measurement_id: str) -> Optional[RealMeasurement]:
        """
        Measure energy for a single prompt.
        
        Args:
            prompt: The prompt to measure
            category: Category of the prompt
            measurement_id: Unique ID for this measurement
            
        Returns:
            RealMeasurement object or None if failed
        """
        if not CODECARBON_AVAILABLE:
            print("CodeCarbon not available!")
            return None
        
        if self.model is None:
            if not self._load_model():
                return None
        
        # Extract features first (outside of measurement)
        features = self._extract_features(prompt)
        
        # Create tracker (offline mode for consistency)
        try:
            tracker = OfflineEmissionsTracker(
                country_iso_code="CAN",  # Canada
                log_level="error",
                save_to_file=False,
            )
        except:
            tracker = EmissionsTracker(
                log_level="error",
                save_to_file=False,
            )
        
        # Measure inference
        try:
            tracker.start()
            output, duration = self._run_inference(prompt)
            emissions = tracker.stop()
            
            # Get energy in kWh (CodeCarbon returns this)
            energy_kwh = tracker._total_energy.kWh if hasattr(tracker, '_total_energy') else emissions / 0.5
            energy_joules = energy_kwh * 3600 * 1000  # Convert kWh to Joules
            carbon_kg = emissions if emissions else 0.0
            
            measurement = RealMeasurement(
                measurement_id=measurement_id,
                prompt=prompt,
                category=category,
                token_count=features["token_count"],
                word_count=features["word_count"],
                char_count=features["char_count"],
                complexity_score=features["complexity_score"],
                avg_word_length=features["avg_word_length"],
                avg_sentence_length=features["avg_sentence_length"],
                model_name=self.model_name,
                model_params_millions=self.model_params_millions,
                energy_kwh=energy_kwh,
                energy_joules=energy_joules,
                duration_seconds=duration,
                carbon_kg=carbon_kg,
                cpu_model=self._get_cpu_info(),
                gpu_available=self._check_gpu(),
                timestamp=pd.Timestamp.now().isoformat(),
            )
            
            return measurement
            
        except Exception as e:
            print(f"Measurement failed for '{prompt[:30]}...': {e}")
            return None
    
    def collect_measurements(self, num_samples: int = 100, verbose: bool = True) -> pd.DataFrame:
        """
        Collect real energy measurements.
        
        Args:
            num_samples: Target number of measurements
            verbose: Print progress
            
        Returns:
            DataFrame with all measurements
        """
        if not CODECARBON_AVAILABLE:
            print("ERROR: CodeCarbon not installed!")
            print("Install with: pip install codecarbon")
            return pd.DataFrame()
        
        # Load model
        if not self._load_model():
            return pd.DataFrame()
        
        # Flatten prompts
        all_prompts = []
        for category, prompts in self.TEST_PROMPTS.items():
            for prompt in prompts:
                all_prompts.append((category, prompt))
        
        # Calculate how many times to repeat
        repeats = max(1, num_samples // len(all_prompts) + 1)
        
        if verbose:
            print(f"\nCollecting {num_samples} real measurements...")
            print(f"Using {len(all_prompts)} unique prompts, {repeats} repeats each")
            print("-" * 50)
        
        measurements = []
        count = 0
        
        for repeat in range(repeats):
            for category, prompt in all_prompts:
                if count >= num_samples:
                    break
                
                measurement_id = f"M{count+1:04d}"
                
                if verbose:
                    print(f"[{count+1}/{num_samples}] {category}: {prompt[:40]}...", end=" ")
                
                measurement = self.measure_single(prompt, category, measurement_id)
                
                if measurement:
                    measurements.append(measurement)
                    if verbose:
                        print(f"✓ {measurement.energy_joules:.4f}J ({measurement.duration_seconds:.3f}s)")
                else:
                    if verbose:
                        print("✗ Failed")
                
                count += 1
                
                # Small delay between measurements
                time.sleep(0.1)
            
            if count >= num_samples:
                break
        
        # Convert to DataFrame
        if measurements:
            df = pd.DataFrame([asdict(m) for m in measurements])
            
            # Save to CSV
            output_path = self.output_dir / "real_measurements.csv"
            df.to_csv(output_path, index=False)
            
            if verbose:
                print("\n" + "=" * 50)
                print(f"Collected {len(df)} measurements")
                print(f"Saved to: {output_path}")
                print("\n--- Statistics ---")
                print(f"Energy range: {df['energy_joules'].min():.4f} - {df['energy_joules'].max():.4f} Joules")
                print(f"Duration range: {df['duration_seconds'].min():.3f} - {df['duration_seconds'].max():.3f} seconds")
                print(f"Token range: {df['token_count'].min()} - {df['token_count'].max()}")
                print(f"\nCorrelations with Energy:")
                numeric_cols = ['token_count', 'word_count', 'complexity_score', 'duration_seconds']
                for col in numeric_cols:
                    corr = df['energy_joules'].corr(df[col])
                    print(f"  {col}: {corr:.3f}")
            
            return df
        
        return pd.DataFrame()


def collect_measurements_fallback(num_samples: int = 100) -> pd.DataFrame:
    """
    Fallback measurement collection using timing-based estimation.
    
    When CodeCarbon isn't available, we use execution time as a proxy
    for energy consumption, with scaling factors from literature.
    
    Energy ≈ Power × Time
    Typical CPU power: 15-65W (laptop) or 65-125W (desktop)
    """
    print("\nUsing timing-based energy estimation (CodeCarbon not available)")
    print("This provides approximate measurements based on execution time.\n")
    
    try:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        print("ERROR: transformers not installed!")
        return pd.DataFrame()
    
    # Load model
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    
    # Estimated power consumption (watts) - conservative laptop estimate
    ESTIMATED_POWER_WATTS = 25.0
    
    # Use prompts with varying lengths (5 to 200+ tokens)
    prompts = [
        # Simple (5-10 tokens)
        ("simple", "What is machine learning?"),
        ("simple", "Define artificial intelligence."),
        ("simple", "Explain Python briefly."),
        # Medium (15-30 tokens)
        ("medium", "How can I improve my Python programming skills effectively and become a better developer?"),
        ("medium", "Explain the differences between supervised learning, unsupervised learning, and reinforcement learning."),
        # Long (50-80 tokens)
        ("long", "Explain how neural networks learn patterns from data through the process of backpropagation, including forward pass, loss calculation, gradient computation, and weight updates. Also describe how different activation functions like ReLU, sigmoid, and tanh affect the learning process."),
        ("long", "Describe the complete process of training a machine learning model from scratch, including data collection, preprocessing, feature engineering, model selection, hyperparameter tuning, cross-validation, training, evaluation, and deployment to production."),
        # Very long (100-150 tokens)
        ("very_long", "Provide a comprehensive analysis of the transformer architecture including self-attention mechanisms, positional encoding, multi-head attention, layer normalization, residual connections, and the encoder-decoder structure. Explain how these components work together for natural language processing tasks like translation, summarization, and question answering. Also discuss the computational complexity and memory requirements of transformers compared to RNNs and LSTMs."),
        ("very_long", "Develop a detailed strategy for building a production-ready machine learning pipeline considering all stages from data ingestion and validation, through feature engineering and selection, model training with cross-validation, hyperparameter optimization using grid search or Bayesian methods, model evaluation with appropriate metrics, deployment using containerization and orchestration, monitoring for data drift and model degradation, and establishing feedback loops for continuous improvement."),
        # Extra long (200+ tokens) - Extended prompts to ensure we test the full token range
        ("extra_long", "Create an exhaustive comparison of different deep learning frameworks including TensorFlow, PyTorch, JAX, and MXNet across multiple dimensions such as ease of use and learning curve, computational performance and optimization capabilities, distributed training support, mobile and edge deployment options, community support and ecosystem, debugging and profiling tools, and production deployment considerations. Provide specific examples and use cases where each framework excels. Additionally discuss the trade-offs between static and dynamic computation graphs and how this affects model development and debugging. Compare the performance of these frameworks on common benchmarks like ImageNet classification and GLUE language understanding tasks. Discuss the impact of hardware accelerators like GPUs and TPUs on framework choice and how each framework handles automatic mixed precision training for improved throughput."),
        ("extra_long", "Analyze the complete lifecycle of a machine learning project from initial problem formulation and stakeholder requirements gathering, through data acquisition and quality assessment, exploratory data analysis and visualization, feature engineering and selection using statistical methods and domain knowledge, model selection comparing different algorithm families like linear models, tree-based ensembles, neural networks, and support vector machines, hyperparameter optimization strategies including random search, grid search, and Bayesian optimization, model evaluation with cross-validation and proper test set holdout, error analysis and debugging, model interpretation using SHAP values and LIME, deployment strategies for batch and real-time inference, monitoring for performance degradation and data drift, and establishing processes for model retraining and versioning. Also cover the importance of reproducibility through version control of data, code, and models, documentation best practices, and establishing clear communication channels between data scientists and stakeholders throughout the project."),
        ("extra_long", "Provide an in-depth technical explanation of how large language models like GPT and BERT work, starting from the fundamental concepts of word embeddings and contextual representations, through the transformer architecture with its self-attention mechanism that allows tokens to attend to all other tokens in the sequence, the role of positional encoding in capturing sequence order information, how multi-head attention enables the model to focus on different aspects of the input simultaneously, the importance of layer normalization and residual connections for stable training of deep networks, the pre-training objectives like masked language modeling for BERT and causal language modeling for GPT, fine-tuning strategies for downstream tasks, techniques for efficient inference like quantization and pruning, and recent advances like instruction tuning and reinforcement learning from human feedback. Additionally explain the scaling laws that govern the relationship between model size, dataset size, and compute budget, and how these findings have influenced the development of increasingly larger models."),
    ]
    
    # Repeat to get enough samples
    all_prompts = prompts * (num_samples // len(prompts) + 1)
    all_prompts = all_prompts[:num_samples]
    
    measurements = []
    
    for i, (category, prompt) in enumerate(all_prompts):
        print(f"[{i+1}/{num_samples}] {prompt[:40]}...", end=" ")
        
        # Tokenize
        input_text = f"summarize: {prompt}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Measure time
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=100, num_beams=4)
        duration = time.perf_counter() - start
        
        # Estimate energy: E = P × t (in Joules)
        energy_joules = ESTIMATED_POWER_WATTS * duration
        energy_kwh = energy_joules / (3600 * 1000)
        
        # Extract features
        words = prompt.split()
        
        measurements.append({
            "measurement_id": f"M{i+1:04d}",
            "prompt": prompt,
            "category": category,
            "token_count": len(tokenizer.encode(prompt)),
            "word_count": len(words),
            "char_count": len(prompt),
            "complexity_score": 0.5,
            "avg_word_length": np.mean([len(w) for w in words]),
            "avg_sentence_length": len(words),
            "model_name": model_name,
            "model_params_millions": 60.0,
            "energy_kwh": energy_kwh,
            "energy_joules": energy_joules,
            "duration_seconds": duration,
            "carbon_kg": energy_kwh * 0.4,  # Approximate carbon intensity
            "cpu_model": "estimated",
            "gpu_available": torch.cuda.is_available(),
            "timestamp": pd.Timestamp.now().isoformat(),
        })
        
        print(f"✓ {energy_joules:.4f}J ({duration:.3f}s)")
        time.sleep(0.05)
    
    df = pd.DataFrame(measurements)
    
    # Save
    output_dir = Path(__file__).parent.parent.parent / "data" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "real_measurements.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nCollected {len(df)} measurements (timing-based estimation)")
    print(f"Saved to: {output_path}")
    
    return df


def collect_measurements(num_samples: int = 100) -> pd.DataFrame:
    """
    Main entry point to collect real measurements.
    
    Uses CodeCarbon if available, otherwise falls back to timing-based estimation.
    """
    if CODECARBON_AVAILABLE:
        collector = RealMeasurementCollector()
        return collector.collect_measurements(num_samples)
    else:
        return collect_measurements_fallback(num_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect real energy measurements")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of measurements")
    args = parser.parse_args()
    
    df = collect_measurements(args.num_samples)
    
    if len(df) > 0:
        print("\n✓ Data collection complete!")
    else:
        print("\n✗ Data collection failed!")
