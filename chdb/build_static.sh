# save the command to files for debug
echo ${LIBCHDB_CMD} > libchdb_cmd.sh.origin
LIBCHDB_CMD=$(gawk '{for(i=1;i<NF;i++){if ($i !~ /\.o$/){printf "%s ", $i}}}END{printf "\n"}' libchdb_cmd.sh.origin)
# generate libchdb.a
object_list=$(gawk '{for(i=0;i<NF;i++){if ($i ~ /\.o$/){printf "%s ", $i}}}END{printf "\n"}' libchdb_cmd.sh.origin)
LIBCHDB_STATIC_CMD=$(echo "llvm-ar-16 rsc libchdb.a ${object_list}")
echo ${LIBCHDB_STATIC_CMD} > libchdb_static.sh
# generate chdb.a
temp_working_dir=$(mktemp --directory)
#------ generate script to create chdb.a begin--------
cat << EOF > libchdb_combine_static.sh
function extract_whole_file {
  ar --output \$2 -x \$1
  for object in \$(ar -t \${1})
  do
    mv "\$2/\${object}" "\$2/\${3}_\${object}" 2>/dev/null || :
  done
}

set -e
rm -f chdb.a || :
echo ${temp_working_dir}
for f in libchdb.a $(echo ${LIBCHDB_CMD} | gawk '{for(i=1;i<NF;i++){if ($i ~ /\.a$/){printf "%s ", $i}}}END{printf "\n"}')
do
  prefix=\$(echo \${f} | sed 's#/#_#g')
  extract_whole_file \${f} ${temp_working_dir} \${prefix}
  ar -t \${f} | awk -v origin_file=\${f} -v dest_dir="${temp_working_dir}" -v prefix="${temp_working_dir}/\${prefix}" '{n[\$0]++}
END{
  for (i in n) {
    if (n[i] < 2) {
      continue;
    }
    printf "cp %s %s\n", origin_file, prefix
    dest_name=prefix"_"i
    extract_name=dest_dir"/"i
    printf "ar --output %s -x %s %s\n", dest_dir, prefix, i
    printf "ar -d %s %s\n", prefix, i
    printf "mv %s %s\n", extract_name, dest_name
    for (j=1; j < n[i]; j++) {
      dest_name=prefix"_"j"_"i
      printf "ar --output %s -x %s %s\n", dest_dir, prefix, i
      printf "ar -d %s %s\n", prefix, i
      printf "mv %s %s\n", extract_name, dest_name
    }
  }
}
  ' | bash # -x # debug
  rm -f "${temp_working_dir}/\${prefix}" || :
  #ar -t ${f} > "/tmp/${prefix}.list"
done
llvm-ar-16 -csr chdb.a ${temp_working_dir}/*.o
ls -lh chdb.a
rm -rf ${temp_working_dir}
EOF
#------ generate script to create chdb.a   end--------

LIBCHDB_CMD=$(echo ${LIBCHDB_CMD} | gawk '{for(i=1;i<NF;i++){if ($i !~ /\.a$/){printf "%s ", $i; if ($i ~ /_chdb.cpython-3.-x86_64-linux-gnu.so/){printf "chdb.a "}}}}END{printf "\n"}')
