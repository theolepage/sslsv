source_path="."
target_path="jeanzay:~/sslsv"

rsync -azh $source_path $target_path \
    --progress \
    --force \
    --exclude=".git" \

while inotifywait -r -e modify,create,delete $source_path
do
    rsync -azh $source_path $target_path \
          --progress \
          --force \
          --exclude=".git"
done
