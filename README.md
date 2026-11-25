# National_energy_security_BN
1. Overview
This supplementary package provides the data, Python code, and Netica network file used in the empirical analysis of energy security presented in this study. These materials support the replication of data preprocessing, Bayesian network construction, sensitivity analysis, and scenario-conditioned inference reported in the manuscript.
The package contains the following files:
•	Supplementary_data.xlsx: Contains the empirical data used to construct and analyse the energy security index, compiled from the U.S. Energy Information Administration (EIA), the Statistical Review of World Energy, and CEIC. The file contains five sheets: Bibliographic Data, Raw Data, Normalized Range, Discretized Data, and Descriptive Statistics.
•	Bayesian_ESI.py: Implements the data preprocessing, discretization, and Bayesian network structure-learning procedures described in the manuscript. The resulting network structure and estimated parameters are stored in Forecast_ESI.neta.
•	Forecast_ESI.neta: A Netica network workspace file containing the learned Bayesian network, including its structure and parameterized conditional probability tables (CPTs). The meanings of all variable abbreviations used in Forecast_ESI.neta are provided in Table 1 below.

Table 1. Variable abbreviations and full names used in the Netica workspace
Abbreviation	Full name	Abbreviation	Full name
EP	Electricity price	CHHI	Consumption HHI
OP	Oil price	IOC	Import to consumption
RTP	Reserve to production	CPC	Carbon emission per capita
WGI	Worldwide Governance Indicators	POC	Production to consumption
CPS	Clean energy production share	CCS	Clean energy consumption share
ECPC	Electricity consumption per capita	EDLPG	Electricity distribution losses ratio
COG	Consumption to GDP	PHHI	Production HHI

Data preprocessing and Bayesian network structure learning were conducted in Python, while probabilistic inference and sensitivity analysis were conducted in Netica.

2. Reproducing the Sensitivity Analysis
The sensitivity analysis reported in the manuscript is based on the joint probability distribution implied by the learned DAG and posterior CPTs.
The discretized observations required for the analysis are provided in the Discretized Data sheet of Supplementary_data.xlsx. The full sample, as well as observations for individual countries and time periods, can be selected and used as evidence in the Netica network.
The procedure is as follows:
•	Identify the country and period of interest, or select the full sample, in the Discretized Data sheet.
•	Extract the corresponding discrete states of the relevant indicator(s).
•	Open Forecast_ESI.neta in Netica.
•	Enter the selected states as evidence for the corresponding nodes.
•	Run probabilistic inference.
•	Obtain the resulting posterior distribution of ESI.
•	Obtain the mutual-information-based importance of the relevant parent nodes.
•	Because the learned network structure and posterior CPTs are embedded in Forecast_ESI.neta, the probabilistic inference and sensitivity analysis can be reproduced without rebuilding the Bayesian network.

3. Scenario-Conditioned Forecasting
The Netica network can also be used to conduct the scenario-conditioned forecasting of ESI reported in the manuscript.
For the forward-looking analysis, scenario-specific values of the explanatory indicators are entered as external evidence in the corresponding network nodes. Netica then propagates this evidence through the network and updates the posterior probability distribution of ESI.
To reproduce a given scenario:
•	Open Forecast_ESI.neta in Netica.
•	Identify the indicator nodes corresponding to the scenario assumptions.
•	Enter the corresponding discrete states as evidence.
•	Run probabilistic inference.
•	Repeat the procedure for alternative scenarios or years as required.
The resulting ESI distribution depends on the scenario-specific indicator values supplied as evidence. The Netica file therefore provides the computational framework for reproducing the scenario-conditioned inference and for testing alternative scenario assumptions.
