from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
import os
import json
import base64
from io import BytesIO

app = Flask(__name__, static_folder='static')
CORS(app)  # 允许跨域请求

# 配置基础路径
BASE_PATH = "./data_multipage_filter"  # 图片文件夹所在的根目录


@app.route('/')
def index():
    """返回前端页面"""
    return send_from_directory('static', 'index.html')


@app.route('/api/folders', methods=['GET'])
def get_folders():
    """获取所有包含PNG文件的文件夹"""
    folders = []
    try:
        for item in os.listdir(BASE_PATH):
            path = os.path.join(BASE_PATH, item)
            if os.path.isdir(path):
                # 获取所有PNG文件
                png_files = [f for f in os.listdir(path) if f.lower().endswith('.png')]
                if png_files:
                    # 按文件名排序
                    png_files.sort()
                    folders.append({
                        'name': item,
                        'images': png_files,
                        'page_count': len(png_files)
                    })
        return jsonify({
            'success': True,
            # 'folders': sorted(folders, key=lambda x: x['name'])
            'folders': sorted(folders, key=lambda x: int(x['name']))
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/image/<folder>/<int:page>', methods=['GET'])
def get_image(folder, page):
    """获取指定文件夹的指定索引的图片"""
    try:
        folder_path = os.path.join(BASE_PATH, folder)

        # 获取所有PNG文件并排序
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        png_files.sort()

        if page < 0 or page >= len(png_files):
            return jsonify({
                'success': False,
                'error': 'Image index out of range'
            }), 404

        img_path = os.path.join(folder_path, png_files[page])

        if not os.path.exists(img_path):
            return jsonify({
                'success': False,
                'error': 'Image not found'
            }), 404

        # 读取图片并转换为base64
        with Image.open(img_path) as img:
            # 转换为RGB（如果是RGBA）
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            # 转换为base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_base64}',
                'width': img.width,
                'height': img.height,
                'filename': png_files[page]
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/save', methods=['POST'])
def save_annotations():
    """保存标注数据"""
    try:
        data = request.json
        folder = data.get('folder')
        page_index = data.get('page')
        filename = data.get('filename')
        rectangles = data.get('rectangles', [])

        if not folder or page_index is None or not rectangles or not filename:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400

        # 获取文件名（不含扩展名）作为保存目录名
        filename_base = os.path.splitext(filename)[0]

        # 创建保存目录
        save_dir = os.path.join(BASE_PATH, folder, filename_base)
        os.makedirs(save_dir, exist_ok=True)

        # 加载原始图片
        img_path = os.path.join(BASE_PATH, folder, filename)
        original_img = Image.open(img_path)

        # 保存裁剪的图片和坐标
        coords_dict = {}
        for i, rect in enumerate(rectangles, 1):
            x1, y1, x2, y2 = rect['x1'], rect['y1'], rect['x2'], rect['y2']

            # 裁剪图片
            cropped = original_img.crop((x1, y1, x2, y2))
            crop_path = os.path.join(save_dir, f'{i}.png')
            cropped.save(crop_path)

            # 记录坐标
            coords_dict[str(i)] = [x1, y1, x2, y2]

        # 保存带标注的完整图片
        from PIL import ImageDraw, ImageFont
        marked_img = original_img.copy()
        draw = ImageDraw.Draw(marked_img)

        for i, rect in enumerate(rectangles, 1):
            x1, y1, x2, y2 = rect['x1'], rect['y1'], rect['x2'], rect['y2']

            # 绘制红色矩形框
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

            # 绘制半透明填充
            overlay = Image.new('RGBA', original_img.size, (255, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 25))
            marked_img.paste(overlay, (0, 0), overlay)

            # 绘制标签文字
            try:
                # 尝试使用默认字体
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except:
                try:
                    # Windows字体
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    # 使用默认字体
                    font = ImageFont.load_default()

            label = f"区域 {i}"
            draw.text((x1 + 5, y1 + 5), label, fill='red', font=font)

        # 保存带标注的完整图片
        marked_path = os.path.join(save_dir, f'{filename_base}_marked.png')
        if marked_img.mode == 'RGBA':
            marked_img = marked_img.convert('RGB')
        marked_img.save(marked_path)

        # 保存JSON文件
        json_path = os.path.join(save_dir, 'coordinates.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coords_dict, f, indent=2, ensure_ascii=False)

        return jsonify({
            'success': True,
            'message': f'Successfully saved {len(rectangles)} regions',
            'save_path': save_dir,
            'marked_image': marked_path
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/annotations/<folder>/<int:page>', methods=['GET'])
def get_annotations(folder, page):
    """获取指定页面的已保存标注"""
    try:
        folder_path = os.path.join(BASE_PATH, folder)

        # 获取所有PNG文件并排序
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        png_files.sort()

        if page < 0 or page >= len(png_files):
            return jsonify({
                'success': False,
                'message': 'Page index out of range'
            })

        # 获取文件名（不含扩展名）
        filename = png_files[page]
        filename_base = os.path.splitext(filename)[0]

        # 检查是否存在保存的标注
        annotations_dir = os.path.join(BASE_PATH, folder, filename_base)
        json_path = os.path.join(annotations_dir, 'coordinates.json')

        if not os.path.exists(json_path):
            return jsonify({
                'success': False,
                'message': 'No saved annotations found'
            })

        # 读取坐标文件
        with open(json_path, 'r', encoding='utf-8') as f:
            coords_dict = json.load(f)

        # 转换为矩形列表
        rectangles = []
        for key in sorted(coords_dict.keys(), key=lambda x: int(x)):
            coords = coords_dict[key]
            rectangles.append({
                'x1': coords[0],
                'y1': coords[1],
                'x2': coords[2],
                'y2': coords[3]
            })

        return jsonify({
            'success': True,
            'rectangles': rectangles
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/pages/<folder>', methods=['GET'])
def get_page_count(folder):
    """获取指定文件夹的页面数量"""
    try:
        folder_path = os.path.join(BASE_PATH, folder)
        if not os.path.exists(folder_path):
            return jsonify({
                'success': False,
                'error': 'Folder not found'
            }), 404

        # 获取所有PNG文件并排序
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        png_files.sort()

        return jsonify({
            'success': True,
            'images': png_files,
            'count': len(png_files)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # 创建static文件夹
    os.makedirs('static', exist_ok=True)

    print("=" * 60)
    print("🚀 服务器启动信息")
    print("=" * 60)
    print(f"📁 图片根目录: {os.path.abspath(BASE_PATH)}")
    print(f"🌐 访问地址: http://localhost:7881")
    print(f"📝 请将前端 HTML 文件保存到 static/index.html")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=7881)