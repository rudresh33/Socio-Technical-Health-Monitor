\# Socio-Technical Project Health Monitor



\*\*Institution:\*\* Goa Business School, Goa University  

\*\*Programme:\*\* M.Sc. Integrated (Data Science) - Semester VI  

\*\*Team:\*\* Unnat Umarye, Samuel Bhandari, Sarvadhnya Patil, Harsh Palyekar, Rudresh Achari  



\## Project Overview

This project proposes a \*\*Socio-Technical Health Monitoring System\*\* that predicts early project delay by integrating structured execution data (JIRA) with sentiment-aware communication analysis (Apache Developer Mailing Lists).



\## System Architecture (Scripts)

\* `data\_acquisition\_api.py`: Automated harvesting of Apache mailing list archives.

\* `entity\_linking\_parser.py`: Regex-based mapping of unstructured emails to structured JIRA task IDs.

\* `dataset\_merger.py`: Aggregation and deduplication of batch-processed data.

\* `feature\_engineering.py`: Creation of temporal metrics (e.g., Days to Resolve, Task Stalled Status).

\* `nlp\_sentiment\_model.py`: VADER sentiment scoring with custom IT-domain lexicon overrides.

\* `eda\_visualizations.py`: Exploratory Data Analysis generating proof-of-concept charts.

