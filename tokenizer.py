import sys, re

infile = sys.argv[1]
outfile = sys.argv[2]


with open(infile, 'r') as f:
	final_str=''
	for line in f:
		newline = line.replace("\'", "")
		newline = newline.replace("/", "")
		newline = newline.replace("/", "")
		newline = newline.replace(".", "")

		#newline= re.sub(r'\'', '', line)

		newline= re.findall('[a-zA-Z0-9]+', newline)
		
		newline= " ".join(newline)
		final_str+=newline
		final_str+='\n'


with open(outfile,'w+') as w:
	w.write(final_str)