# Experiment: Analyzing Historical BGP Event

This experiment demonstrates how BEAM models can be used to evaluate the path differences caused by route changes, highlighting that anomalous route changes typically exhibit significantly greater path differences compared to legitimate ones. It also shows how to detect historical BGP events and generate corresponding reports and diagrams.

## Prerequisites

Before proceeding, please follow the instructions in the main `readme.md` to set up your environment.

## Get Started

First, run `experiment/init.sh` to:

1. Download pre-trained embeddings (or, you can train the embeddings yourself).
2. Download a pre-compiled BGPdump tool (or, you can compile it yourself).
3. Download the report template file.
4. Download ground-truth anomalous updates (or you have to modify the code to incorporate inputs from all RouteViews collectors to ensure visibility of all events).
5. Download the expected outputs for reference.

Next, run `experiment/run.py` to iterate over all known events. For each event, the script will:

1. Fetch the routing data surrounding the event time.
2. Detect all route changes within the time span.
3. Extract the ground truth (legitimate/anomalous route changes).
4. Evaluate the path differences of the ground-truth route changes, generate a report, and plot a CDF diagram.

See all available parameters with `--help`.

Examples:

```bash
# Run all events at once to save your trouble:
python run.py 

# Or run a specific event:
python run.py --collector wide \
              --ev-code pakistan
```

See the `expected_output` directory for the expected results. 

Note: This process requires downloading a significant amount of data, so please ensure you have a stable Internet connection.

---
