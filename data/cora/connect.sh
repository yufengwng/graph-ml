#!/usr/bin/env bash
#
# See https://relational.fit.cvut.cz/dataset/CORA.

HOST='relational.fit.cvut.cz'
PORT=3306
USER='guest'
PASS='relational'
DB='CORA'

set -x
mycli --host $HOST --port $PORT -u $USER --pass $PASS $DB
