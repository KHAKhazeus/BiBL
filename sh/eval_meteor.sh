# Replace xxx with real data paths. java env is required. meteor should be cloned into the project path and successfully setup.
java -Xmx2G -jar ./meteor-1.5/meteor-1.5.jar ./data/xxx/pred.text.txt.tok ./data/xxx/gold.text.txt.tok -l en -norm > ./data/xxx/meteor_metrics