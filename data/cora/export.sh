#!/usr/bin/env bash

HOST='relational.fit.cvut.cz'
PORT=3306
USER='guest'
PASS='relational'
DB='CORA'

tables=(paper content cites)
for table in ${tables[@]}; do
  cmd="select * from $table;"
  echo "exporting table [$table] ..."
  mariadb --host=$HOST --port=$PORT --user=$USER --password=$PASS $DB \
    -e "$cmd" | sed 's/\t/,/g' > "$table.csv"
done
