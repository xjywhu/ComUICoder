from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import json
import shutil
from pathlib import Path

app = Flask(__name__)

# 配置根目录路径
# ROOT_DIR = './data_multipage_filter'  # 修改为你的实际路径
ROOT_DIR = './data'  # 修改为你的实际路径

def get_folders():
    """获取所有文件夹列表"""
    folders = []
    if os.path.exists(ROOT_DIR):
        for item in os.listdir(ROOT_DIR):
            item_path = os.path.join(ROOT_DIR, item)
            if os.path.isdir(item_path):
                folders.append(item)
        # 按照数字顺序排序
        folders.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))

    return folders


def get_groups(folder_name):
    """获取指定文件夹下的所有 group"""
    groups = {}
    group_path = os.path.join(ROOT_DIR, folder_name, 'group')

    if os.path.exists(group_path):
        group_list = []
        for group_name in os.listdir(group_path):
            group_dir = os.path.join(group_path, group_name)
            if os.path.isdir(group_dir):
                group_list.append(group_name)

        # 按照数字顺序排序 group
        group_list.sort(
            key=lambda x: int(x.replace('group', '')) if x.startswith('group') and x[5:].isdigit() else float('inf'))

        print(group_list)

        for group_name in group_list:
            group_dir = os.path.join(group_path, group_name)
            images = []
            for img in sorted(os.listdir(group_dir)):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append({
                        'name': img,
                        'path': f'/image/{folder_name}/group/{group_name}/{img}'
                    })
            if images:
                groups[group_name.replace("group", "")] = images

    return groups


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/folders')
def api_folders():
    """获取文件夹列表"""
    folders = get_folders()
    return jsonify(folders)


@app.route('/api/groups/<folder_name>')
def api_groups(folder_name):
    """获取指定文件夹的 groups"""
    groups = get_groups(folder_name)
    return jsonify(groups)


@app.route('/api/merge', methods=['POST'])
def api_merge():
    """合并选中的图片到新 group"""
    data = request.json
    folder_name = data.get('folder')
    selected_images = data.get('images', [])

    if not selected_images:
        return jsonify({'error': '没有选中的图片'}), 400

    # 创建新 group
    group_path = os.path.join(ROOT_DIR, folder_name, 'group')
    existing_groups = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]

    # 生成新的 group 名称
    group_numbers = []
    for g in existing_groups:
        if g.startswith('group'):
            try:
                num = int(g.replace('group', ''))
                group_numbers.append(num)
            except:
                pass

    new_group_num = max(group_numbers) + 1 if group_numbers else 1
    new_group_name = f'group{new_group_num}'
    new_group_path = os.path.join(group_path, new_group_name)

    # 创建新 group 目录
    os.makedirs(new_group_path, exist_ok=True)

    # 移动选中的图片到新 group，并删除原 group 中的图片
    groups_to_check = {}
    for img_info in selected_images:
        if "group" not in img_info['group']:
            img_info['group'] = "group" + img_info['group']
        source_path = os.path.join(ROOT_DIR, folder_name, 'group',
                                   img_info['group'], img_info['image'])
        dest_path = os.path.join(new_group_path, img_info['image'])

        if os.path.exists(source_path):
            # 移动文件而不是复制
            shutil.move(source_path, dest_path)

            # 记录需要检查的 group
            source_group_path = os.path.join(ROOT_DIR, folder_name, 'group', img_info['group'])
            groups_to_check[source_group_path] = True

    # 检查并删除空的 group 文件夹
    for group_dir in groups_to_check.keys():
        if os.path.exists(group_dir):
            remaining_files = [f for f in os.listdir(group_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not remaining_files:
                shutil.rmtree(group_dir)

    return jsonify({
        'success': True,
        'new_group': new_group_name,
        'message': f'成功创建 {new_group_name} 并合并了 {len(selected_images)} 张图片'
    })


@app.route('/api/renumber', methods=['POST'])
def api_renumber():
    """重新编号所有 group 文件夹"""
    data = request.json
    folder_name = data.get('folder')

    group_path = os.path.join(ROOT_DIR, folder_name, 'group')
    if not os.path.exists(group_path):
        return jsonify({'error': 'group 文件夹不存在'}), 400

    # 获取所有 group 文件夹
    groups = []
    for item in os.listdir(group_path):
        item_path = os.path.join(group_path, item)
        if os.path.isdir(item_path) and item.startswith('group'):
            try:
                num = int(item.replace('group', ''))
                groups.append((num, item))
            except:
                pass

    # 按照数字排序
    groups.sort(key=lambda x: x[0])

    # 创建临时目录
    temp_path = os.path.join(ROOT_DIR, folder_name, 'group_temp')
    os.makedirs(temp_path, exist_ok=True)

    # 先移动到临时目录
    for i, (old_num, old_name) in enumerate(groups, 1):
        old_path = os.path.join(group_path, old_name)
        temp_group_path = os.path.join(temp_path, f'group{i}')
        shutil.move(old_path, temp_group_path)

    # 再移回 group 目录
    for item in os.listdir(temp_path):
        src = os.path.join(temp_path, item)
        dst = os.path.join(group_path, item)
        shutil.move(src, dst)

    # 删除临时目录
    shutil.rmtree(temp_path)

    return jsonify({
        'success': True,
        'message': f'成功重新编号 {len(groups)} 个 group'
    })


@app.route('/image/<path:filepath>')
def serve_image(filepath):
    """提供图片文件"""
    return send_from_directory(ROOT_DIR, filepath)


if __name__ == '__main__':
    # 创建 templates 目录
    # os.makedirs('templates', exist_ok=True)
    #
    # with open('templates/index.html', 'w', encoding='utf-8') as f:
    #     f.write(html_content)

    print("服务器启动中...")
    print("请在浏览器中访问: http://localhost:7881")
    print(f"数据目录: {os.path.abspath(ROOT_DIR)}")

    app.run(debug=True, host='0.0.0.0', port=7881)