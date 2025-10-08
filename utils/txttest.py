def read_file(file_path):
    """读取文件内容并返回内容字符串"""
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def compare_files(file1_path, file2_path):
    """比较两个文件的内容是否一致"""
    content1 = read_file(file1_path)
    content2 = read_file(file2_path)

    if content1 == content2:
        return True
    else:
        return False

# 指定两个文件的路径

file1_path = '/home/ztb/ABD/test/cifar/data_2.txt'
file2_path = '/home/ztb/bench/record/badnet_0_1/bd_test_dataset/9/txt/data_2.txt'

# 比较两个文件的内容
if compare_files(file1_path, file2_path):
    print("两个文件的内容相同")
else:
    print("两个文件的内容不同")