project(CMGLab_STG_Presentation NONE)
cmake_minimum_required(VERSION 3.19)
SET(LATEX_OUTPUT_PATH build)
SET(PDFLATEX_COMPILER xelatex)
set(LATEX_COMPILER_FLAGS
"-interaction=nonstopmode -file-line-error -shell-escape"
CACHE STRING "Flags passed to latex."
)
INCLUDE(UseLATEX.cmake)

# file(GLOB_RECURSE contents_files RELATIVE ${CMAKE_SOURCE_DIR} source/*.tex)

ADD_LATEX_DOCUMENT(
    main.tex
    # INPUTS ${contents_files}
    IMAGE_DIRS img
    )
