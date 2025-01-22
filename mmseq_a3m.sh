mmseqs easy-search fastas/all5.fasta ../databases/uniref90/uniref90db_pad ./mmseqs2_output/all5.a3m tmp --gpu 1 --split-mode 2 --format-mode 4

mmseqs result2msa tmp/10539281033120586044/query ../databases/uniref90/uniref90db_pad tmp/10539281033120586044/result ./mmseqs2_output/A7FASTA.a3m --msa-format-mode 5
