import sys

file1 = sys.argv[1]

with open(file1, 'r') as f:
    lCount = 0
    for line in f:
        lCount += 1
        if lCount > 100:
            break
        list_line = list(line)
        # print(list_line)

        present_word_length = 0
        for i in range(len(list_line)):
            if list_line[i] == '\n':
                pass
            elif list_line[i] == ' ':
                with open('out.csv', 'a') as out:
                    st = list_line[i] + ',' + 'O' + '\n'
                    out.write(st)
                out.close()
            else:
                if present_word_length == 0 and (list_line[i + 1] == ' ' or list_line[i + 1] == '\n'):
                    with open('out.csv', 'a') as out:
                        st = list_line[i] + ',' + 'S' + '\n'
                        out.write(st)
                    out.close()
                elif present_word_length == 0 and list_line != ' ':
                    present_word_length += 1
                    with open('out.csv', 'a') as out:
                        st = list_line[i] + ',' + 'B' + '\n'
                        out.write(st)
                    out.close()
                elif present_word_length != 0 and list_line[i + 1] != ' ':
                    present_word_length += 1
                    with open('out.csv', 'a') as out:
                        st = list_line[i] + ',' + 'I' + '\n'
                        out.write(st)
                    out.close()
                elif present_word_length != 0 and (list_line[i + 1] == ' ' or list_line[i + 1] == '\n'):
                    present_word_length = 0
                    with open('out.csv', 'a') as out:
                        st = list_line[i] + ',' + 'E' + '\n'
                        out.write(st)
                    out.close()

f.close()
