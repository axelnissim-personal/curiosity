# 🧠 Curiosity-Driven Knowledge Exploration System

## Overview

This project is a curiosity-driven knowledge exploration system named **Curiosity**, inspired by the concepts outlined in the paper [Curiosity Is All You Need: A Comprehensive Framework for Intrinsically Motivated Language Models](https://github.com/axelnissim-personal/curiosity/blob/main/Curiosity_Is_all_you_need.md). 

The system autonomously explores concepts and relationships, building a dynamic knowledge graph. It uses a combination of probabilistic reasoning, reinforcement learning, and natural language processing to generate hypotheses, refine its understanding, and produce a reflective narrative of its intellectual journey. The output is presented in an HTML report, which includes a visual representation of the knowledge graph, a list of hypotheses, and a creative essay.

The system is designed to be highly observable, addressing the opacity of traditional Large Language Models (LLMs) by providing transparency into its reasoning process. Each hypothesis is annotated with confidence scores, frequency metrics, and relationship classifications, making the system's outputs interpretable and trustworthy.

## Features

### 🔍 Curiosity-Driven Exploration
- The system autonomously explores concepts and relationships, driven by curiosity scores that balance novelty, uncertainty, and reward
- It generates hypotheses about relationships between concepts and validates them using an LLM

### 🕸️ Dynamic Knowledge Graph
- The system builds and updates a knowledge graph that visually represents the relationships between concepts
- Each node (concept) and edge (relationship) is annotated with metadata such as confidence scores, frequency, and classification types

### 🧪 Hypothesis Tracking
- The system tracks and numbers each hypothesis, providing a clear audit trail of its reasoning process
- Hypotheses are classified into types (e.g., Supertype-Subtype, Causal, Functional, etc.), adding semantic context to the knowledge graph

### 📝 Reflective Narrative
- The system generates a creative essay that reflects on its intellectual journey, tying together hypotheses and insights
- This narrative provides a human-readable summary of the system's exploration

### 📊 HTML Report
The system produces an HTML report that includes:
- A visual representation of the knowledge graph
- A list of seed concepts and their relationships
- A detailed summary of hypotheses, including confidence scores and classifications
- A creative essay that narrates the system's exploration

### 👁️ Observability
- The system is highly observable, providing transparency into its reasoning process
- This makes the system's outputs interpretable and trustworthy, addressing the opacity of traditional LLMs

## Prototype Nature and Constraints

### Prototype Showcasing a Concept
This system is a prototype designed to showcase the power of curiosity-driven learning and dynamic knowledge graph construction. It demonstrates how AI can autonomously explore and reason about abstract concepts, generate hypotheses, and reflect on its own learning process.

### Artificial Constraint: 45 Hypotheses
To make the prototype manageable and focused, the system is artificially constrained to generate a maximum of 45 hypotheses. This constraint ensures that the exploration process is concise and that the output can be easily analyzed and interpreted.

### Infinite Learning Potential
Unlike traditional LLMs, which are constrained to a fixed dataset and a specific point in time, this system has the potential to learn infinitely and in a greedy manner. It can continuously update its knowledge graph in real time, incorporating new information and refining its understanding as it explores.

### Real-Time Learning
The system is designed to learn in real time, updating its knowledge graph on the fly as new hypotheses are generated and validated. This makes it a powerful tool for dynamic and evolving domains, where new information is constantly being discovered.

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Colab (optional, for running the script in a notebook environment)
- A Google API key for accessing the Gemini LLM

### Install Dependencies
Run the following command to install the required Python packages:

```bash
pip install ratelimit networkx torch sentence-transformers google-generativeai matplotlib
```

### Set Up Google API Key
1. Obtain a Google API key from the Google Cloud Console
2. Store the API key in Google Colab's userdata module or as an environment variable

## Usage

### Running the Script

1. **Import the Script**:
   - If running in Google Colab, upload the script to your Colab environment
   - If running locally, save the script as `curiosity_system.py`

2. **Set Up Initial Concepts**:
   - Modify the `initial_concepts` list in the script to include the seed concepts and relationships you want the system to explore

   Example:
   ```python
   initial_concepts = [
       ("Consciousness", {"involves": "Subjective Experience"}),
       ("Neural Activity", {"correlates with": "Conscious States"}),
       ("Curiosity", {"involves": "Intrinsic Motivation"}),
       ("Absolute certainty", {"is contrary to": "Curiosity"})
   ]
   ```

3. **Run the Script**:
   - Execute the script in your preferred environment (Google Colab or local Python interpreter)

   Example:
   ```bash
   python curiosity_system.py
   ```

4. **View the Output**:
   - The script will generate an HTML report named `Curiosity{timestamp}.html`
   - Open the report in a web browser to explore the system's findings

## Output

The system produces an HTML report that includes the following sections:

### Knowledge Graph Diagram
- A visual representation of the knowledge graph, with nodes representing concepts and edges representing relationships
- Seed concepts are highlighted in blue, and new concepts are highlighted in green

### Seed Concepts
- A list of the initial concepts and relationships provided to the system

### Most Recent Worldview
- A detailed summary of hypotheses, including confidence scores, frequency metrics, and relationship classifications

### Creative Essay
- A narrative generated by the system, reflecting on its intellectual journey and the insights gained from exploring the hypotheses

## Customization

### Modifying Seed Concepts
To explore different domains or topics, modify the `initial_concepts` list in the script. Each entry in the list should be a tuple containing a concept and a dictionary of relationships.

Example:
```python
initial_concepts = [
    ("Quantum Mechanics", {"is related to": "Particle Physics"}),
    ("Artificial Intelligence", {"involves": "Machine Learning"}),
    ("Climate Change", {"causes": "Global Warming"})
]
```

### Adjusting Exploration Parameters
You can adjust the following parameters to customize the system's exploration:

- **Top N Concepts**: The number of high-curiosity concepts to explore in each cycle
  ```python
  ui.curiosity.explore(top_n=3)
  ```

- **Hypothesis Limit**: The maximum number of hypotheses to generate
  ```python
  if ui.curiosity.hypothesis_counter >= 45:
      break
  ```

- **Learning Interval**: The time interval between autonomous learning cycles
  ```python
  self.learning_interval = 10  # Seconds
  ```

## Examples

### Example 1: Exploring Consciousness
**Seed Concepts**:
```python
initial_concepts = [
    ("Consciousness", {"involves": "Subjective Experience"}),
    ("Neural Activity", {"correlates with": "Conscious States"}),
    ("Curiosity", {"involves": "Intrinsic Motivation"}),
    ("Absolute certainty", {"is contrary to": "Curiosity"})
]
```

**Output**:
- The system explores relationships between consciousness, neural activity, curiosity, and absolute certainty
- It generates hypotheses about how these concepts interact and produces a reflective essay on its findings

### Example 2: Exploring Climate Change
**Seed Concepts**:
```python
initial_concepts = [
    ("Climate Change", {"causes": "Global Warming"}),
    ("Greenhouse Gases", {"contribute to": "Climate Change"}),
    ("Renewable Energy", {"mitigates": "Climate Change"})
]
```

**Output**:
- The system explores the causes and effects of climate change, generating hypotheses about the role of greenhouse gases and renewable energy
- It produces a narrative that reflects on the challenges and solutions related to climate change

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Commit your changes and push them to your fork
4. Submit a pull request with a detailed description of your changes

## License

This project is licensed under the GPL 3.0 License. See the LICENSE file for details.

## Acknowledgments

- Google Generative AI: For providing the Gemini LLM used in this project
- NetworkX: For the knowledge graph construction and visualization
- Sentence-Transformers: For generating concept embeddings
- Matplotlib: For creating the knowledge graph diagram

## Logo

The system's logo is a female cat, symbolizing curiosity and exploration. The cat represents the system's drive to seek out new knowledge and its playful, inquisitive nature.

## About the Author

Axel Nissim is the creator of the Curiosity system. Inspired by the paper "Curiosity Is All You Need: A Comprehensive Framework for Intrinsically Motivated Language Models", Axel built this prototype to demonstrate the power of curiosity-driven learning and dynamic knowledge graph construction. You can reach Axel at axelnissim@gmail.com.
