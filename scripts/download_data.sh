#!/bin/bash
# Download forensic sketch datasets (CUFS primary, CUFSF harder).
#
# CUHK mmlab servers are frequently down; this script tries the direct
# academic mirrors first and prints clear manual fallback instructions
# (Kaggle / academic mirrors) if they fail. Nothing in here is fatal:
# every download is wrapped so the script always exits 0 and tells you
# what to do next.
set -u

DATA_DIR="data"
mkdir -p "$DATA_DIR/raw"

echo "=== Downloading CUHK Face Sketch (CUFS) ==="
# CUFS is available via the CUHK mmlab page. Direct mirrors also exist on
# Kaggle and various academic mirrors.
wget -q -O "$DATA_DIR/raw/cufs.zip" \
  "http://mmlab.ie.cuhk.edu.hk/archive/facesketch.zip" \
  && echo "  -> CUFS downloaded to $DATA_DIR/raw/cufs.zip" \
  || echo "  !! CUFS direct download failed — see MANUAL FALLBACK below"

echo "=== Downloading CUFSF (FERET-based, harder) ==="
wget -q -O "$DATA_DIR/raw/cufsf.zip" \
  "http://mmlab.ie.cuhk.edu.hk/datasets/face_sketch_FERET_database/cufsf.zip" \
  && echo "  -> CUFSF downloaded to $DATA_DIR/raw/cufsf.zip" \
  || echo "  !! CUFSF direct download failed — see MANUAL FALLBACK below"

cat <<'EOF'

=========================== MANUAL FALLBACK ===========================
If either download above failed (CUHK servers are often offline), use one
of these mirrors and unzip into the expected layout:

  data/cufs/sketch/<id>.png   data/cufs/photo/<id>.jpg
  data/cufsf/sketch/<id>.png  data/cufsf/photo/<id>.jpg

The sketch and its matching photo MUST share the same filename stem so the
preprocessing pairing step can match them.

1) Kaggle (requires `pip install kaggle` and an API token in ~/.kaggle):

     kaggle datasets download -d arbazkhan971/cuhk-face-sketch-database-cufs \
       -p data/raw/
     unzip -o data/raw/cuhk-face-sketch-database-cufs.zip -d data/raw/cufs_kaggle

   Then sort the extracted sketches/photos into data/cufs/sketch and
   data/cufs/photo with matching stems.

2) Reference repo with download notes + a pix2pix baseline:
     https://github.com/pratheeshkumar99/SketchGAN

3) CUFSF source page:
     http://mmlab.ie.cuhk.edu.hk/datasets/face_sketch_FERET_database/
=======================================================================
EOF

echo "Done. Verify pairs exist under data/cufs/ before running preprocessing."
