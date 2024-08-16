# Semantics-Aware Routing Anomaly Detection System

This document provides an overview of the routing anomaly detection system and details the configuration requirements for both the training and detection phases of the system.

Please refer to the [paper](https://www.usenix.org/conference/usenixsecurity24/presentation/chen-yihao) for full details.

Contact: yh-chen21@mails.tsinghua.edu.cn

## System Overview

<img src="doc/detection-prototype.png">

As shown in the diagram, the routing anomaly detection system consists of three main modules:

-   **Routing Monitor Module:** Takes multi-collector routing update announcements as input and outputs detected routing changes in real-time.
-   **BEAM Engine Module:** Uses AS business relationship data as input to train the BEAM network representation learning model offline, which is used to assess the path difference (abnormality) of routing changes.
-   **Anomaly Detector Module:** Performs real-time anomaly detection on the routing changes identified by the Routing Monitor using the trained BEAM model and conducts correlation analysis on detected anomalous routing changes, outputting anomaly alerts.

## Training Phase

Training is performed offline using the latest AS business relationship data to obtain the BEAM network representation learning model. The trained model is then used during real-time detection.

-   **Training Input:** AS business relationship data. For example, [CAIDA AS Relationship Data](https://publicdata.caida.org/datasets/as-relationships/) is updated monthly. An example archive (20230201.as-rel2.txt.bz2) contains 501,844 AS relationship records and is approximately 1.5MB in size.

    The BEAM engine can automatically access CAIDA's database to fetch the relevant data. Alternatively, other sources can be used if they follow the same format.

-   **Training Output:** Three sets of binary files containing the trained embedding vectors (link.emb, node.emb, rela.emb), with a total size of approximately 50MB.

-   **Environment Requirements:** Python 3.6, necessary Python libraries (numpy 1.19.4, pandas 1.1.4, pytorch 1.3.1), and a GPU environment is recommended (cuda 11.7).

-   **Configuration and Performance Reference:** On a dual-core Xeon E5-2650v4 with a GeForce RTX 2080 Ti, the training time is about 10 hours. Memory usage does not exceed 10GB, and disk space usage is less than 1GB.

## Detection Phase

Once the model training is complete, the system uses multi-collector real-time routing update announcements for routing anomaly detection.

-   **Detection Input:** Multi-collector routing update announcements. For example, [RouteViews Routing Updates](http://routeviews.org/) maintains over 30 collectors, each generating multi-collector routing updates in [MRT format](https://www.rfc-editor.org/rfc/rfc6396) at approximately 15-minute intervals. The routing monitor module can automatically fetch and process this data using the bgpdump tool. Other sources can also be used if they adhere to the MRT format.

-   **Detection Output:** Anomaly alert files containing the time window, prefix, associated AS, and corresponding anomalous routing changes.

-   **Environment Requirements:** Python >=3.6, necessary Python libraries (numpy 1.19.4, pandas 1.1.4, scipy 1.5.4, tqdm 4.52.0, joblib 0.13.0), and gcc toolchain for compiling bgpdump.

-   **Configuration and Performance Reference:** On a dual-core Xeon E5-2650v4, the routing monitor module processes 15 minutes of data from all RouteViews collectors in approximately 140 seconds, while the anomaly detector processes 1,000 routing changes in about 0.05 seconds. At least 30GB of memory is recommended, and disk space usage primarily depends on the size of the input data (it is advised to reserve twice the input data size).

---
