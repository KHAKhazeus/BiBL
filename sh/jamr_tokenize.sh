# create a tokenizer.sh within jamr to complete the tokenization process
rm -rf ./jamr_tokenize.log; ./jamr/scripts/tokenizer.sh > ./jamr_tokenize.log 2>&1;