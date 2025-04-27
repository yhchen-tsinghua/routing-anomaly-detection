#!/usr/bin/env bash

REPO_DIR=`cd $(dirname $0)/..; pwd`

# download pre-trained embeddings
if [ ! -f $REPO_DIR/BEAM_engine/models.zip ]; then
    cd $REPO_DIR/BEAM_engine
    curl -L -o models.zip https://cloud.tsinghua.edu.cn/f/263b7e6388964a6c82c8/?dl=1
    unzip models.zip
    echo "Download pre-trained embeddings to $(pwd)/models/"
fi

# download pre-compiled BGPdump
if [ ! -f $REPO_DIR/data/routeviews/bgpd ]; then
    cd $REPO_DIR/data/routeviews
    curl -L -o bgpd https://cloud.tsinghua.edu.cn/f/611648a5aacd473daab8/?dl=1
    chmod +x bgpd
    echo "Download BGPdump tool to $(pwd)/bgpd"
fi

# download report template
if [ ! -f $REPO_DIR/experiment/report_template.html ]; then
    cd $REPO_DIR/experiment
    curl -L -o report_template.html https://cloud.tsinghua.edu.cn/f/10d02517e67449ba8acb/?dl=1
    echo "Download report template to $(pwd)/report_template.html"
fi

# download ground-truth anomalous updates
if [ ! -f $REPO_DIR/experiment/anomaly_gt.zip ]; then
    cd $REPO_DIR/experiment
    curl -L -o anomaly_gt.zip https://cloud.tsinghua.edu.cn/f/7c8b3ac0f05141de86d6/?dl=1
    unzip anomaly_gt.zip
    echo "Download ground-truth updates to $(pwd)/anomaly_gt/"
fi

# download benchmark results
if [ ! -f $REPO_DIR/experiment/anomaly_gt.zip ]; then
    cd $REPO_DIR/experiment
    curl -L -o anomaly_gt.zip https://cloud.tsinghua.edu.cn/f/7c8b3ac0f05141de86d6/?dl=1
    unzip anomaly_gt.zip
    echo "Download ground-truth updates to $(pwd)/anomaly_gt/"
fi

# download expected output
if [ ! -f $REPO_DIR/experiment/expected_output.zip ]; then
    cd $REPO_DIR/experiment
    curl -L -o expected_output.zip https://cloud.tsinghua.edu.cn/f/2fa718f3f59142f2af4b/?dl=1
    unzip expected_output.zip
    echo "Download expected outputs to $(pwd)/expected_output/"
fi

echo "Initialized."
