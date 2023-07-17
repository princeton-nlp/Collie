mkdir -p data
wget -P ./data/ http://static.decontextualize.com/gutenberg-dammit-files-v002.zip
unzip -qo data/gutenberg-dammit-files-v002.zip -d ./data
# wget -P ./data/ https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt
wget -P ./data/ http://www.gwicks.net/textlists/english3.zip
unzip -qo data/english3.zip -d ./data