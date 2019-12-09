rm -rf .data/squad/
mkdir .data/squad
cd data_preprocess/
python generate_json.py
mv data/json/* ../.data/squad/
cd ..
python run.py --train-file test.json --dev-file dev.json --train-batch-size 10 --dev-batch-size 10

