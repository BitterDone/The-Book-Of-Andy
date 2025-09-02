#!/usr/bin/env bash
echo "cd transcripts"
cd transcripts
echo "cd'd transcripts. make removeSpecialChars"
mkdir removeSpecialChars
echo "made removeSpecialChars. move txt files"
mv *.txt ./removeSpecialChars/
echo "moved txt files. cd removeSpecialChars"
cd removeSpecialChars/

echo "cd'd removeSpecialChars. do for loop"
for file in *;
do
    mv "$file" ../$(echo "$file" | sed -e 's/[^A-Za-z0-9._-]/_/g'); done;

echo "did for loop. cd out of removeSpecialChars"
cd ..
echo "cd'd out of removeSpecialChars. rm removeSpecialChars"
rm -rf removeSpecialChars
echo "rm'd removeSpecialChars. make cutLeadingPrefix"
mkdir cutLeadingPrefix
echo "made cutLeadingPrefix. mv txt to cutLeadingPrefix"
mv *.txt cutLeadingPrefix/
echo "mv'd txt to cutLeadingPrefix. cd cutLeadingPrefix"

cd cutLeadingPrefix/
echo "cd'd cutLeadingPrefix. do 2nd for"

for file in *;
do
    prefix="${file:0:13}";
    filename="${file:13}";
    base="${filename%.txt}";
    newname="${base}_._${prefix}.txt";
    mv "$file" ../"$newname"; 
done

echo "did 2nd for. cd out of cutLeadingPrefix"
cd ..
echo "cd'd out of cutLeadingPrefix. rm cutLeadingPrefix"
rm -rf cutLeadingPrefix
echo "rm'd cutLeadingPrefix. done"
ls -al transcripts



# prefix="hell"
# suffix="ld"
# string="hello-world"
# foo=${string#"$prefix"}
# foo=${foo%"$suffix"}

# echo "${foo}"
#     o-wor