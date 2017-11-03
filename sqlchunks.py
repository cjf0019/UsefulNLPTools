# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:34:38 2017

@author: InfiniteJest
"""

def chunkize_query(chunksize, data):
    fmt =  '''
            '''

    for chunk in (data[i:i + chunksize] for i in range(0, len(data), chunksize)):
        q = fmt.format(column_data = ', '.join(["'"+str(i)+"'" for i in chunk]))
        print('M FOR MINI')
        yield q

def chunkize_query_percent(no_chunks, percent):
    chunk = percent/no_chunks
    fmt = '''

Where (ABS(CAST((BINARY_CHECKSUM(*) * RAND()) as int)) % 100) < {amount}    
         '''
    chunk_run = 0
    while chunk_run < percent:
        q = fmt.format(amount = chunk)
        print('W FOR WUMBO')
        chunk_run += chunk
        yield q

def write_sqlchunks_to_file(file_name, chunks, with_header=True, delimiter=',',quotechar='"',quoting=csv.QUOTE_NONNUMERIC):
    ofile = open(file_name,'a', newline='', encoding='utf-8')
    csv_writer = csv.writer(ofile, delimiter=delimiter, quotechar=quotechar,quoting=quoting)
    cur = conn.cursor()    
    cur.execute(next(chunks))
    if with_header:
        column = [field[0] for field in cur.description]
        csv_writer.writerow(column)
    first_results = cur.fetchall()
    for result in first_results:
        csv_writer.writerow(result)
    cur.close()
    for chunk in chunks:
        cur = conn.cursor()
        cur.execute(chunk)
        results = cur.fetchall()
        for result in results:
            csv_writer.writerow(result)
        cur.close()
    ofile.close()
