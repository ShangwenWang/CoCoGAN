import json
import os

for language in ['java', 'python']:
    print(language)
    train,valid,test,codebase=[],[],[], []
    for root, dirs, files in os.walk(language+'/final'):
        for file in files:
            temp=os.path.join(root,file)
            if '.jsonl' in temp:
                if 'train' in temp:
                    train.append(temp)
                elif 'valid' in temp:
                    valid.append(temp)
                    codebase.append(temp)
                elif 'test' in temp:
                    test.append(temp) 
                    codebase.append(temp)
                    
    train_data,valid_data,test_data,codebase_data={},{},{},{}
    for files,data in [[train,train_data],[valid,valid_data],[test,test_data],[codebase,codebase_data]]:
            for file in files:
                if '.gz' in file:
                    os.system("gzip -d {}".format(file))
                    file=file.replace('.gz','')
                with open(file) as f:
                    for line in f:
                        line=line.strip()
                        js=json.loads(line)
                        data[js['url']]=js
                        