from __future__ import absolute_import, division, print_function
# Python
from six.moves import range, map
import os
import re
import textwrap
try:
    import numpy as np
except ImportError:
    pass
from os.path import join, splitext
# Util
from utool import util_cplat
from utool import util_path
from utool import util_num
from utool import util_dev
from utool import util_io
from utool.util_dbg import printex
from utool.util_inject import inject
print, print_, printDBG, rrr, profile = inject(__name__, '[latex]')

#def ensure_latex_environ():
#    paths = os.environ['PATH'].split(os.pathsep)
#    mpl.rc('font',**{'family':'serif'})
#    mpl.rc('text', usetex=True)
#    mpl.rc('text.latex',unicode=True)
#    mpl.rc('text.latex',preamble='\usepackage[utf8]{inputenc}')


def find_ghostscript_exe():
    import utool as ut
    if ut.WIN32:
        gs_exe = r'C:\Program Files (x86)\gs\gs9.16\bin\gswin32c.exe'
    else:
        gs_exe = 'gs'
    return gs_exe


def compress_pdf(pdf_fpath, output_fname=None):
    """ uses ghostscript to write a pdf """
    import utool as ut
    ut.assertpath(pdf_fpath)
    suffix = '_' + ut.get_datestamp(False) + '_compressed'
    print('pdf_fpath = %r' % (pdf_fpath,))
    output_pdf_fpath = ut.augpath(pdf_fpath, suffix, newfname=output_fname)
    print('output_pdf_fpath = %r' % (output_pdf_fpath,))
    gs_exe = find_ghostscript_exe()
    cmd_list = (
        gs_exe,
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        '-sOutputFile=' + output_pdf_fpath,
        pdf_fpath
    )
    ut.cmd(*cmd_list)
    return output_pdf_fpath


def make_full_document(text):
    import utool as ut
    doc_preamb = ut.codeblock(r'''
    %\documentclass{article}
    \documentclass[10pt,twocolumn,letterpaper]{article}

    \usepackage[english=nohyphenation]{hyphsubst}
    \usepackage{times}
    \usepackage{amsmath}
    \usepackage{amsthm}
    \usepackage{amssymb}
    \usepackage[usenames,dvipsnames,svgnames,table]{xcolor}
    \usepackage{mathtools}
    \usepackage{algorithm}
    \usepackage{ulem}
    \normalem
    \usepackage{graphicx,adjustbox}
    \usepackage{multirow}
    \usepackage[T1]{fontenc}
    \usepackage[margin=1.25in]{geometry}

    %\pagenumbering{gobble}
    ''')
    doc_header = ut.codeblock(r'''
    \begin{document}
    ''')
    doc_footer = ut.codeblock(r'''
    \end{document}
    ''')
    text_ = '\n'.join((doc_preamb, doc_header, text, doc_footer))
    return text_


def compile_latex_text(input_text, fnum=1, dpath=None, verbose=True):
    """
    pdflatex -shell-escape --synctex=-1 -src-specials -interaction=nonstopmode /home/joncrall/code/ibeis/tmptex/latex_formatter_temp.tex

    Example1:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> import utool as ut
        >>> verbose = True
        >>> dpath = '/home/joncrall/code/ibeis/aidchallenge'
        >>> ut.vd(dpath)
        >>> orig_fpath_list = ut.list_images(dpath, fullpath=True)
        >>> figure_str = ut.get_latex_figure_str(new_rel_fpath_list, width_str='2.4in', nCols=2)
        >>> input_text = figure_str
        >>> pdf_fpath = ut.compile_latex_text(input_text, dpath=dpath, verbose=verbose)
        >>> output_pdf_fpath = ut.compress_pdf(pdf_fpath)

        fpath_list
        def clipwhite_ondisk(fpath_in):
            import utool as ut
            import vtool as vt
            fpath_out = ut.augpath(fpath_in, '_clipwhite')
            img = vt.imread(fpath_in)
            thresh = 128
            fillval = [255, 255, 255]
            cropped_img = vt.crop_out_imgfill(img, fillval=fillval, thresh=thresh)
            vt.imwrite(fpath_out, cropped_img)
            return fpath_out
        fpath_list_ = [clipwhite_ondisk(fpath) for fpath in fpath_list]
        tmpfig = join(dpath, 'tmpfig')
        ut.ensuredir(tmpfig)
        # Weirdness
        from os.path import *
        new_fpath_list = []
        for fpath in fpath_list_:
            fname, ext = splitext(basename(fpath))
            fname_ = ut.hashstr(fname, alphabet=ut.ALPHABET_16) + ext
            fpath_ = join(tmpfig, fname_)
            ut.move(fpath, fpath_)
            new_fpath_list.append(fpath_)
        new_rel_fpath_list = [ut.relpath_unix(fpath_, dpath) for fpath_ in new_fpath_list]

    """
    #import pylab as plt
    #import matplotlib as mpl
    #verbose = True
    text = make_full_document(input_text)
    cwd = os.getcwd()
    if dpath is None:
        text_dir = join(cwd, 'tmptex')
    else:
        text_dir = dpath
    util_path.ensuredir(text_dir, verbose=verbose)
    text_fname = 'latex_formatter_temp.tex'
    text_fpath = join(text_dir, text_fname)
    pdf_fpath = splitext(text_fpath)[0] + '.pdf'
    #jpg_fpath = splitext(text_fpath)[0] + '.jpg'
    try:
        os.chdir(text_dir)
        util_io.write_to(text_fpath, text)
        pdflatex_args = ('pdflatex', '-shell-escape', '--synctex=-1', '-src-specials', '-interaction=nonstopmode')
        args = pdflatex_args + (text_fpath,)
        util_cplat.cmd(*args, verbose=verbose)
        assert util_path.checkpath(pdf_fpath, verbose=verbose), 'latex failed'
    except Exception as ex:
        ut.printex(ex, 'LATEX ERROR')
    finally:
        os.chdir(cwd)
    return pdf_fpath


def render(input_text, fnum=1, dpath=None, verbose=True):
    import pylab as plt
    import matplotlib as mpl
    #verbose = True
    text = make_full_document(input_text)
    cwd = os.getcwd()
    if dpath is None:
        text_dir = join(cwd, 'tmptex')
    else:
        text_dir = dpath
    util_path.ensuredir(text_dir, verbose=verbose)
    text_fname = 'latex_formatter_temp.tex'
    text_fpath = join(text_dir, text_fname)
    pdf_fpath = splitext(text_fpath)[0] + '.pdf'
    jpg_fpath = splitext(text_fpath)[0] + '.jpg'
    try:
        os.chdir(text_dir)
        util_io.write_to(text_fpath, text)
        pdflatex_args = ('pdflatex', '-shell-escape', '--synctex=-1', '-src-specials', '-interaction=nonstopmode')
        args = pdflatex_args + (text_fpath,)
        util_cplat.cmd(*args, verbose=verbose)
        assert util_path.checkpath(pdf_fpath, verbose=verbose), 'latex failed'
        # convert latex pdf to jpeg
        util_cplat.cmd('convert', '-density', '300', pdf_fpath, '-quality', '90', jpg_fpath, verbose=verbose)
        assert util_path.checkpath(jpg_fpath, verbose=verbose), 'imgmagick failed'
        tex_img = plt.imread(jpg_fpath)
        # Crop img bbox
        nonwhite_x = np.where(tex_img.flatten() != 255)[0]
        nonwhite_rows = nonwhite_x // tex_img.shape[1]
        nonwhite_cols = nonwhite_x % tex_img.shape[1]
        x1 = nonwhite_cols.min()
        y1 = nonwhite_rows.min()
        x2 = nonwhite_cols.max()
        y2 = nonwhite_rows.max()
        #util.embed()
        cropped = tex_img[y1:y2, x1:x2]
        fig = plt.figure(fnum)
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cropped, cmap=mpl.cm.gray)
        #mpl.rc('text', usetex=True)
        #mpl.rc('font', family='serif')
        #plt.figure()
        #plt.text(9, 3.4, text, size=12)
    except Exception as ex:
        print('LATEX ERROR')
        print(text)
        print(ex)
        print('LATEX ERROR')
        pass
    finally:
        os.chdir(cwd)
        #if dpath is None:
        #    if util_path.checkpath(text_dir, verbose=verbose):
        #        util_path.delete(text_dir)


def latex_multicolumn(data, ncol=2):
    data = escape_latex(data)
    return r'\multicolumn{%d}{|c|}{%s}' % (ncol, data)


def latex_multirow(data, nrow=2):
    return r'\multirow{%d}{*}{|c|}{%s}' % (nrow, data)


def latex_get_stats(lbl, data, mode=0):
    stats_ = util_dev.get_stats(data)
    if stats_.get('empty_list', False):
        return '% NA: latex_get_stats, data=[]'
    try:
        max_ = stats_['max']
        min_ = stats_['min']
        mean = stats_['mean']
        std  = stats_['std']
        shape = stats_['shape']
    except KeyError as ex:
        stat_keys = stats_.keys()  # NOQA
        printex(ex, key_list=['stat_keys', 'stats_', 'data'])
        raise

    #int_fmt = lambda num: util.num_fmt(int(num))
    float_fmt = lambda num: util_num.num_fmt(float(num))
    tup_fmt = lambda tup: str(tup)
    fmttup = (float_fmt(min_), float_fmt(max_), float_fmt(mean), float_fmt(std), tup_fmt(shape))
    lll = ' ' * len(lbl)
    if mode == 0:
        prefmtstr = r'''
        {label} stats & min ; max = %s ; %s\\
        {space}       & mean; std = %s ; %s\\
        {space}       & shape = %s \\'''
    if mode == 1:
        prefmtstr = r'''
        {label} stats & min  = $%s$\\
        {space}       & max  = $%s$\\
        {space}       & mean = $%s$\\
        {space}       & std  = $%s$\\
        {space}       & shape = $%s$\\'''
    fmtstr = prefmtstr.format(label=lbl, space=lll)
    latex_str = textwrap.dedent(fmtstr % fmttup).strip('\n') + '\n'
    return latex_str


def latex_scalar(lbl, data):
    return (r'%s & %s\\' % (lbl, util_num.num_fmt(data))) + '\n'


def make_stats_tabular():
    'tabular for dipslaying statistics'
    pass


def ensure_rowvec(arr):
    arr = np.array(arr)
    arr.shape = (1, arr.size)
    return arr


def ensure_colvec(arr):
    arr = np.array(arr)
    arr.shape = (arr.size, 1)
    return arr


def padvec(shape=(1, 1)):
    pad = np.array([[' ' for c in range(shape[1])] for r in range(shape[0])])
    return pad


def escape_latex(unescaped_latex_str):
    ret = unescaped_latex_str
    ret = ret.replace('#', '\\#')
    ret = ret.replace('%', '\\%')
    ret = ret.replace('_', '\\_')
    return ret


def replace_all(str_, repltups):
    ret = str_
    for ser, rep in repltups:
        ret = re.sub(ser, rep, ret)
    return ret


def make_score_tabular(row_lbls, col_lbls, scores, title=None,
                       out_of=None, bold_best=True,
                       replace_rowlbl=None, flip=False):
    """ tabular for displaying scores """
    bigger_is_better = True
    if flip:
        bigger_is_better = not bigger_is_better
        flip_repltups = [('<', '>'), ('score', 'error')]
        col_lbls = [replace_all(lbl, flip_repltups) for lbl in col_lbls]
        if title is not None:
            title = replace_all(title, flip_repltups)
        if out_of is not None:
            scores = out_of - scores

    if replace_rowlbl is not None:
        for ser, rep in replace_rowlbl:
            row_lbls = [re.sub(ser, rep, lbl) for lbl in row_lbls]

    # Abbreviate based on common substrings
    SHORTEN_ROW_LBLS = True
    common_rowlbl = None
    if SHORTEN_ROW_LBLS:
        if isinstance(row_lbls, list):
            row_lbl_list = row_lbls
        else:
            row_lbl_list = row_lbls.flatten().tolist()
        # Split the rob labels into the alg components
        #algcomp_list = [lbl.split(')_') for lbl in row_lbl_list]
        longest = long_substr(row_lbl_list)
        common_strs = []
        while len(longest) > 10:
            common_strs += [longest]
            row_lbl_list = [row.replace(longest, '...') for row in row_lbl_list]
            longest = long_substr(row_lbl_list)
        common_rowlbl = ('...'.join(common_strs)).replace(')_', ')_\n')
        row_lbls = row_lbl_list
        if len(row_lbl_list) == 1:
            common_rowlbl = row_lbl_list[0]
            row_lbls = ['0']

    # Stack into a tabular body
    col_lbls = ensure_rowvec(col_lbls)
    row_lbls = ensure_colvec(row_lbls)
    _0 = np.vstack([padvec(), row_lbls])
    _1 = np.vstack([col_lbls, scores])
    body = np.hstack([_0, _1])
    body = [[str_ for str_ in row] for row in body]
    # Fix things in each body cell
    AUTOFIX_LATEX = True
    FORCE_INT = True
    DO_PERCENT = True
    for r in range(len(body)):
        for c in range(len(body[0])):
            # In data land
            if r > 0 and c > 0:
                # Force integer
                if FORCE_INT:
                    body[r][c] = str(int(float(body[r][c])))
            # Remove bad formatting;
            if AUTOFIX_LATEX:
                body[r][c] = escape_latex(body[r][c])

    # Bold the best scores
    if bold_best:
        best_col_scores = scores.max(0) if bigger_is_better else scores.min(0)
        rows_to_bold = [np.where(scores[:, colx] == best_col_scores[colx])[0]
                        for colx in range(len(scores.T))]
        for colx, rowx_list in enumerate(rows_to_bold):
            for rowx in rowx_list:
                body[rowx + 1][colx + 1] = '\\txtbf{' + body[rowx + 1][colx + 1] + '}'

    # More fixing after the bold is in place
    for r in range(len(body)):
        for c in range(len(body[0])):
            # In data land
            if r > 0 and c > 0:
                if out_of is not None:
                    body[r][c] = body[r][c] + '/' + str(out_of)
                    if DO_PERCENT:
                        percent = ' = %.1f%%' % float(100 * scores[r - 1, c - 1] / out_of)
                        body[r][c] += escape_latex(percent)

    # Align columns for pretty printing
    body = np.array(body)
    ALIGN_BODY = True
    if ALIGN_BODY:
        new_body_cols = []
        for col in body.T:
            colstrs = list(map(str, col.tolist()))
            collens = list(map(len, colstrs))
            maxlen = max(collens)
            newcols = [str_ + (' ' * (maxlen - len(str_))) for str_ in colstrs]
            new_body_cols += [newcols]
        body = np.array(new_body_cols).T

    # Build Body (and row layout)
    HLINE_SEP = True
    rowsep = ''
    colsep = '&'
    endl = '\\\\\n'
    hline = r'\hline'
    extra_rowsep_pos_list = [1]
    if HLINE_SEP:
        rowsep = hline + '\n'
    rowstr_list = [colsep.join(row) + endl for row in body]
    # Insert title
    if title is not None:
        tex_title = latex_multicolumn(title, len(body[0])) + endl
        rowstr_list = [tex_title] + rowstr_list
        extra_rowsep_pos_list += [2]
    # Apply an extra hline (for label)
    for pos in sorted(extra_rowsep_pos_list)[::-1]:
        rowstr_list.insert(pos, '')
    tabular_body = rowsep.join(rowstr_list)

    # Build Column Layout
    col_layout_sep = '|'
    col_layout_list = ['l'] * len(body[0])
    extra_collayoutsep_pos_list = [1]
    for pos in  sorted(extra_collayoutsep_pos_list)[::-1]:
        col_layout_list.insert(pos, '')
    col_layout = col_layout_sep.join(col_layout_list)

    tabular_head = (r'\begin{tabular}{|%s|}' % col_layout) + '\n'
    tabular_tail = r'\end{tabular}'

    tabular_str = rowsep.join([tabular_head, tabular_body, tabular_tail])

    if common_rowlbl is not None:
        #tabular_str += escape_latex('\n\nThe following parameters were held fixed:\n' + common_rowlbl)
        pass
    return tabular_str


def get_latex_figure_str(fpath_list, caption_str=None, label_str=None, width_str=r'\textwdith', height_str=None, nCols=None, dpath=None):
    r"""
    Args:
        fpath_list (list):
        dpath (str): directory relative to main tex file

    Returns:
        str: figure_str

    CommandLine:
        python -m utool.util_latex --test-get_latex_figure_str

    Example:
        >>> # DISABLE_DOCTEST
        >>> from utool.util_latex import *  # NOQA
        >>> # build test data
        >>> fpath_list = ['figures/foo.png']
        >>> # execute function
        >>> figure_str = get_latex_figure_str(fpath_list)
        >>> # verify results
        >>> result = str(figure_str)
        >>> print(result)
    """
    import utool as ut
    if width_str is not None:
        graphics_sizestr = '[width=%s]' % (width_str)
    elif height_str is not None:
        graphics_sizestr = '[height=%s]' % (height_str)
    else:
        graphics_sizestr =  ''

    if dpath is not None:
        fpath_list = [ut.relpath_unix(fpath_, dpath) for fpath_ in fpath_list]
    graphics_list = [r'\includegraphics%s{%s}' % (graphics_sizestr, fpath,) for fpath in fpath_list]

    if nCols is None:
        nCols = len(graphics_list)
    #nRows = len(graphics_list) // nCols

    # Add separators
    sep_list = [
        '\n&\n' if count % nCols > 0 else '\n\\\\\n'
        for count in range(1, len(graphics_list) + 1)
    ]
    sep_list[-1] = ''
    graphics_list_ = [graphstr + sep for graphstr, sep in zip(graphics_list, sep_list)]

    #graphics_body = '\n&\n'.join(graphics_list)
    graphics_body = ''.join(graphics_list_)
    header_str = ' '.join(['c'] * nCols)

    tabular_body =  ut.codeblock(
        r'''
        \begin{tabular}{%s}
        %s
        \end{tabular}
        '''
    ) % (header_str, graphics_body)
    if caption_str is not None:
        #tabular_body += '\n\caption{\\footnotesize{%s}}' % (caption_str,)
        tabular_body += '\n\caption{%s}' % (caption_str,)
    if label_str is not None:
        tabular_body += '\n\label{fig:%s}' % (label_str,)
    figure_fmtstr = ut.codeblock(
        r'''
        \begin{figure*}
        \begin{center}
        %s
        \end{center}
        \end{figure*}
        '''
    )
    figure_str = figure_fmtstr % (tabular_body)
    return figure_str


def _tabular_header_and_footer(col_layout):
    tabular_head = textwrap.dedent(r'\begin{tabular}{|%s|}' % col_layout)
    tabular_tail = textwrap.dedent(r'\end{tabular}')
    return tabular_head, tabular_tail


def long_substr(strlist):
    # Longest common substring
    # http://stackoverflow.com/questions/2892931/longest-common-substring-from-more-than-two-strings-python
    substr = ''
    if len(strlist) > 1 and len(strlist[0]) > 0:
        for i in range(len(strlist[0])):
            for j in range(len(strlist[0]) - i + 1):
                if j > len(substr) and is_substr(strlist[0][i:i + j], strlist):
                    substr = strlist[0][i:i + j]
    return substr


def is_substr(find, strlist):
    if len(strlist) < 1 and len(find) < 1:
        return False
    for i in range(len(strlist)):
        if find not in strlist[i]:
            return False
    return True


def tabular_join(tabular_body_list, nCols=2):
    dedent = textwrap.dedent
    tabular_head = dedent(r'''
    \begin{tabular}{|l|l|}
    ''')
    tabular_tail = dedent(r'''
    \end{tabular}
    ''')
    hline = ''.join([r'\hline', '\n'])
    tabular_body = hline.join(tabular_body_list)
    tabular = hline.join([tabular_head, tabular_body, tabular_tail])
    return tabular


if __name__ == '__main__':
    """
    CommandLine:
        python -m utool.util_latex
        python -m utool.util_latex --allexamples
        python -m utool.util_latex --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
