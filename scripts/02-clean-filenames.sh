#!/usr/bin/env bash

cd transcripts

mkdir removeSpecialChars
mv *.txt ./removeSpecialChars/
cd removeSpecialChars/

for file in *; do mv "$file" ../$(echo "$file" | sed -e 's/[^A-Za-z0-9._-]/_/g'); done &

cd ..
rm -rf removeSpecialChars

mkdir cutLeadingPrefix
mv *.txt ./cutLeadingPrefix/

# cd cutLeadingPrefix/

# for file in *;
# do
#     prefix="${file:0:13}";
#     extension="${file:0:13}";
#     filename="${file:13}";
#     newname="$filename _._$prefix";
#     mv "$file" ../"$newname"; 
# done &

# cd ..
# rm -rf cutLeadingPrefix




# prefix="hell"
# suffix="ld"
# string="hello-world"
# foo=${string#"$prefix"}
# foo=${foo%"$suffix"}

# echo "${foo}"
#     o-wor