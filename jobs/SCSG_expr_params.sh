#!/bin/bash

python3 ~/utils/pyparams.py -f "SCSG_expr_params.txt" \
    '["adult", "connect", "credit", "crowdsource", "mnist", "sensor"]' \
    '[0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]' \
