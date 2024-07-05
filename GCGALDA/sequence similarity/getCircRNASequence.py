import pandas as pd
import numpy as np

def find_pair_sequence():
    # 读取RNA名称文件
    with open('./2/RNA_names.fa', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 初始化矩阵和列表
    circ_name_list = []
    final_matrix = np.zeros((len(lines), 2), dtype=object)
    final_matrix = pd.DataFrame(final_matrix, columns=['RNA_Name', 'Sequence'])
    
    # 填充矩阵和列表
    index = 0
    for line in lines:
        line = line.strip()
        if line != "":  # 忽略空行
            circ_name_list.append(line)
            final_matrix.iloc[index, 0] = line
            index += 1
    
    # 读取并处理RNA序列文件
    with open('./rna_seq.txt', 'r', encoding='utf-8') as f:
        database_name = ""
        sequence = ""
        found = False
        not_found_count = 0  # 记录未找到匹配的数量
        for line in f:
            line = line.strip()
            if line.startswith(">>"):
                if found:
                    # 如果找到匹配的RNA名称，则保存序列并重置状态
                    if database_name in final_matrix['RNA_Name'].values:
                        final_matrix.loc[final_matrix['RNA_Name'] == database_name, 'Sequence'] = sequence.strip()
                    found = False
                # 提取数据库中的RNA名称
                database_name = line[2:].strip()
                sequence = ""  # 重置序列
                # 检查是否有匹配的RNA名称
                if database_name in circ_name_list:
                    found = True
            elif line.startswith(">"):
                # 次级信息行，忽略
                continue
            elif found:
                # 累积序列
                sequence += line
        
        # 最后一个序列的处理
        if found and database_name in final_matrix['RNA_Name'].values:
            final_matrix.loc[final_matrix['RNA_Name'] == database_name, 'Sequence'] = sequence.strip()
    
    # 删除 RNA_Name 列为0的行
    final_matrix = final_matrix[final_matrix['RNA_Name'] != 0]
    not_found_count = len(final_matrix[final_matrix['Sequence'] == 0])
    
    # 保存结果到Excel文件
    final_matrix.to_excel('my_sequence_information.xlsx', header=True, index=False)
    
    # 打印未找到匹配的数量
    print(f"共有 {not_found_count} 个RNA名称在rna_seq_v2017.txt中未找到匹配。")

# 调用函数
find_pair_sequence()



def getCircName():
    # data = pd.read_excel('data2/circR2Disease-human.xlsx')
    # circ_name = data['circRNA Name'].drop_duplicates(ignore_index=True)
    input_file_path = './2/lncrna_name.txt'
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        disease_names = input_file.readlines()

    # 创建FASTA格式的文件
    with open('2/RNA_names.fa', 'w') as fasta_file:
        for name in disease_names:
            fasta_file.write(f"{name}\n")

    print("FASTA文件已成功创建！")

#getCircName()