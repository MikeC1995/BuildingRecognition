#!/bin/bash
echo "Looking for images in:" $1 "..."
a=1
for i in $1/*.jpg $1/*.JPG; do
  new=$(printf $1/"%04d.jpg" "$a") #04 pad to length of 4
  mv -- "$i" "$new"
  let a=a+1
done
