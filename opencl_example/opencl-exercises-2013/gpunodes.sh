#!/bin/bash
#
# Simple script for displaying load and users logged into GPU nodes on
# BlueCrystal Phase 1.
#
# Written by James Price (j.price@bristol.ac.uk)

echo "Node    GPU1 GPU2     CPU/IO Load     Users"
echo "-------------------------------------------"
for i in {1..6}
do
  echo -n "gpu00$i: "
  ssh gpu00$i 'nvidia-smi -q -d UTILIZATION' | grep Gpu | awk '{printf "%3.0f%% ",$3}'
  ssh gpu00$i 'cat /proc/loadavg' | head -n 1 | awk '{printf "[%5.2f %5.2f %5.2f] ",$1,$2,$3}'
  ssh gpu00$i who | sort -u -k 1,1 | awk '{printf "%s ",$1}'
  echo
done

