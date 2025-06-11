#!/usr/bin/bash

# 10-20-2022
# This script pbtaines gene coordinates for a specified window around a gene or a gene's transcription start site
# Take in gene list and GENCODE annotations, create a new file with this format:
#   Column 1: gene
#   Column 2: region

# to run:
# ./get_gene_coords.sh -g <gene_list> -a <gencode annotations> -r <region type: tss/genebody> -w <window size> -o <output dir>
#
# example:
# ./get_gene_coords.sh -g gene_list.txt -a gencode.v41.annotation.gtf -r genebody -w 100000 -o Output_Direcotory

while getopts "g:a:r:w:o:" option
do
        case $option in
                g) gene_list=$OPTARG;;
                a) annotations=$OPTARG;;
                r) region=$OPTARG;;
                w) window=$OPTARG;;
                o) output=$OPTARG;;
        esac
done

# Begin writing output
mkdir $output

# Get gene coordinates from gencode annotations; make new file with regions

genes="$(awk '{print $1}' $gene_list | sort | uniq)"

if [ $region == "tss" ]
then
        echo "Choosing peaks around TSS"
        echo -e "gene\ttss_window_$window" > $output/genelist_tss.txt
        for gene in $genes
        do
                # get chr and tss
               chr="$(awk '{if ($3=="gene") {print}}' $annotations | sed 's/"//g' | sed 's/;//g' | awk -v g=$gene '{if ($14==g) {print $1}}')"
                # check if gene in file by length of chr
               length="$(echo $chr | wc -c)"
               if [ $length -gt 2 ]
               then
                       echo "Getting window for $gene"
                       tss="$(awk '{if ($3=="gene") {print}}' $annotations | sed 's/"//g' | sed 's/;//g' | awk -v g=$gene '{if ($14==g) {print $4}}')"
                        # compute window around TSS
                       start="$(echo "$(($tss-$window))")"
                       stop="$(echo "$(($tss+$window))")"
        
                       echo -e "$gene\t$chr:$start-$stop" >> $output/genelist_tss.txt
               else
                       echo "$gene not in annotations file"
               fi
        done
elif [ $region == "genebody" ]
then
        echo "Choosing peaks around genebody"
        echo -e "gene\tgenebody_window_$window" > $output/genelist_genebody.txt
                for gene in $genes
        do
                # get chr and tss
                chr="$(awk '{if ($3=="gene") {print}}' $annotations | sed 's/"//g' | sed 's/;//g' | awk -v g=$gene '{if ($14==g) {print $1}}')"
                # check if gene in file by length of chr
                length="$(echo $chr | wc -c)"
                if [ $length -gt 2 ]
                then
                        echo "Getting window for $gene"
                        tss="$(awk '{if ($3=="gene") {print}}' $annotations | sed 's/"//g' | sed 's/;//g' | awk -v g=$gene '{if ($14==g) {print $4}}')"
                        tes="$(awk '{if ($3=="gene") {print}}' $annotations | sed 's/"//g' | sed 's/;//g' | awk -v g=$gene '{if ($14==g) {print $5}}')"
                        # compute window around GENEBODY
                        start="$(echo "$(($tss-$window))")"
                        stop="$(echo "$(($tes+$window))")"

                        echo -e "$gene\t$chr:$start-$stop" >> $output/genelist_genebody.txt
                else
                        echo "$gene not in annotations file"
                fi

        done
else
        echo "Please select 'tss' or 'genebody' for peak window selection"
fi



