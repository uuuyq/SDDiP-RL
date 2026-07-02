#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF to Markdown Converter using Marker
使用 Marker 将 PDF 转换为 Markdown 格式
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 添加 marker 导入
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
except ImportError as e:
    print(json.dumps({
        "success": False,
        "error": f"Failed to import marker: {str(e)}. Please install marker-pdf: pip install marker-pdf"
    }, ensure_ascii=False))
    sys.exit(1)


def log(message):
    """打印带时间戳的日志"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def convert_pdf_to_markdown(pdf_path: str, output_dir: str = None, converter: PdfConverter = None) -> dict:
    """
    将单个 PDF 文件转换为 Markdown

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录（可选）
        converter: 已初始化的 PdfConverter 实例（可选，避免重复加载模型）

    Returns:
        dict: 包含转换结果的字典
    """
    try:
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path}"
            }

        if not pdf_file.suffix.lower() == '.pdf':
            return {
                "success": False,
                "error": f"File is not a PDF: {pdf_path}"
            }

        # 如果没有传入 converter，则创建一个（会加载模型）
        if converter is None:
            log("正在加载 AI 模型（首次加载需要较长时间，请耐心等待）...")
            model_dict = create_model_dict()
            converter = PdfConverter(artifact_dict=model_dict)

        # 转换 PDF
        log(f"正在转换: {pdf_file.name}")
        rendered = converter(str(pdf_file))
        full_text, images, out_meta = text_from_rendered(rendered)

        # 确定输出路径
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = pdf_file.parent

        # 生成输出文件名
        base_name = pdf_file.stem
        md_file = output_path / f"{base_name}.md"

        # 写入 Markdown 文件
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(full_text)

        # 保存提取的图片
        if images:
            img_dir = output_path / f"{base_name}_images"
            img_dir.mkdir(parents=True, exist_ok=True)
            for img_name, pil_img in images.items():
                img_path = img_dir / img_name
                pil_img.save(str(img_path))

        # 返回结果
        result = {
            "success": True,
            "input_file": str(pdf_file),
            "output_file": str(md_file),
            "title": out_meta.get("title", base_name),
            "page_count": out_meta.get("page_count", 0),
            "images_count": len(images),
            "markdown_length": len(full_text)
        }

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "input_file": pdf_path
        }


def batch_convert_pdfs(input_dir: str, output_dir: str = None) -> dict:
    """
    批量转换目录中的所有 PDF 文件

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径（可选，默认为输入目录）

    Returns:
        dict: 包含批量转换结果的字典
    """
    try:
        start_time = time.time()
        input_path = Path(input_dir)

        if not input_path.exists():
            return {
                "success": False,
                "error": f"Input directory not found: {input_dir}"
            }

        # 查找所有 PDF 文件
        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            return {
                "success": False,
                "error": f"No PDF files found in: {input_dir}"
            }

        log(f"找到 {len(pdf_files)} 个 PDF 文件")
        for f in pdf_files:
            log(f"  - {f.name}")

        # 设置输出目录
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            out_path = input_path

        # 加载模型并创建 converter（只加载一次）
        log("正在加载 AI 模型（首次加载需要较长时间，请耐心等待）...")
        model_start = time.time()
        model_dict = create_model_dict()
        converter = PdfConverter(artifact_dict=model_dict)
        model_time = time.time() - model_start
        log(f"模型加载完成，耗时: {model_time:.2f} 秒")

        results = []
        success_count = 0

        for idx, pdf_file in enumerate(pdf_files, 1):
            file_start = time.time()
            log(f"[{idx}/{len(pdf_files)}] 开始转换: {pdf_file.name}")

            try:
                rendered = converter(str(pdf_file))
                full_text, images, out_meta = text_from_rendered(rendered)

                # 生成输出文件名
                base_name = pdf_file.stem
                md_file = out_path / f"{base_name}.md"

                # 写入 Markdown 文件
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(full_text)

                # 保存提取的图片
                if images:
                    img_dir = out_path / f"{base_name}_images"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    for img_name, pil_img in images.items():
                        img_path = img_dir / img_name
                        pil_img.save(str(img_path))

                file_time = time.time() - file_start
                log(f"[{idx}/{len(pdf_files)}] 转换完成: {pdf_file.name} ({file_time:.2f} 秒)")

                results.append({
                    "success": True,
                    "input_file": str(pdf_file),
                    "output_file": str(md_file),
                    "title": out_meta.get("title", base_name),
                    "page_count": out_meta.get("page_count", 0),
                    "images_count": len(images),
                    "time_seconds": file_time
                })
                success_count += 1

            except Exception as e:
                file_time = time.time() - file_start
                log(f"[{idx}/{len(pdf_files)}] 转换失败: {pdf_file.name} - {str(e)} ({file_time:.2f} 秒)")
                results.append({
                    "success": False,
                    "input_file": str(pdf_file),
                    "error": str(e),
                    "time_seconds": file_time
                })

        total_time = time.time() - start_time
        log(f"批量转换完成！总计: {total_time:.2f} 秒")

        return {
            "success": True,
            "total": len(pdf_files),
            "success_count": success_count,
            "failed_count": len(pdf_files) - success_count,
            "total_time_seconds": total_time,
            "results": results
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def main():
    # ========== 参数写死，PyCharm 右键直接执行 ==========
    input_dir = r"d:\tools\workspace_pycharm\SDDiP-RL\modeling\doc"
    output_dir = None  # None 表示输出到输入目录同目录
    # ==================================================

    result = batch_convert_pdfs(input_dir, output_dir)

    # 输出 JSON 结果到 stdout
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
