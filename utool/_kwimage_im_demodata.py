"""
A Port of kwimage.grab_test_image_fpath to replace inflexible
utool.grab_test_imgpath implementation

import liberator
lib = liberator.Liberator()
import kwimage
lib.add_dynamic(kwimage.grab_test_image_fpath)
print(lib.current_sourcecode())

# lib.add_dynamic(ub.grabdata)
# lib.add_dynamic(ub.Cacher)
# lib.add_dynamic(ub.CacheStamp)


# Check PNG encoding

import kwimage
fpaths = []
fpaths += [kwimage.imwrite('data1.png', np.zeros((1, 1, 3), dtype=np.uint8))]
fpaths += [kwimage.imwrite('data2.png', np.zeros((10, 10, 3), dtype=np.uint8))]
fpaths += [kwimage.imwrite('data3.png', np.zeros((100, 100, 3), dtype=np.uint8))]
fpaths += [kwimage.imwrite('data4.png', np.zeros((5120, 5120, 3), dtype=np.uint8))]
for fpath in map(ub.Path, fpaths):
    print(f'fpath = {ub.urepr(fpath, nl=1)}')
    data = fpath.read_bytes()
    print(f'data = {ub.urepr(data, nl=1)}')

data1 = ub.Path('data1.png').read_bytes()
data2 = ub.Path('data2.png').read_bytes()
print(f'data1 = {ub.urepr(data1, nl=1)}')
print(f'data2 = {ub.urepr(data2, nl=1)}')
"""
import ubelt as ub


def _grabdata_with_mirrors(url, mirror_urls, grabkw):
    fpath = None
    verbose = 1
    try:
        fpath = ub.grabdata(url, **grabkw)
    except Exception as main_ex:
        if verbose:
            print(f'Failed to grab main url: {main_ex}')
            print('Fallback to mirrors:')
        # urllib.error.HTTPError
        for idx, mirror_url in enumerate(mirror_urls):
            try:
                fpath = ub.grabdata(mirror_url, **grabkw)
            except Exception as ex:
                if verbose:
                    print(f'Failed to grab mirror #{idx}: {ex}')
            else:
                break
        if fpath is None:
            raise main_ex
    return fpath


_TEST_IMAGES = {
    'airport': {
        'fname': 'airport.jpg',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/9/9e/Beijing_Capital_International_Airport_on_18_February_2018_-_SkySat_%281%29.jpg',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb7ea71cc6eae69303aa/download',
        ],
        'ipfs_cids': [
            'bafkreif76x4sclk4o7oup4vybzo4dncat6tycoyi7q43kbbjisl3gsb77q',
        ],
        'sha256': 'bff5f9212d5c77dd47f2b80e5dc1b4409fa7813b08fc39b504294497b3483ffc',
        'sha512': '957695b319d8b8266e4eece845b016fbf2eb4f1b6889d7374d29ab812f752da77e42a6cb664bf15cc38face439bd60070e85e5b7954be15fc354b07b353b9582',
        'properties': {
            'shape': (
                868,
                1156,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'amazon': {
        'fname': 'amazon.jpg',
        'url': 'https://data.kitware.com/api/v1/file/611e9f4b2fa25629b9dc0ca2/download',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb85a71cc6eae69303ad/download',
        ],
        'ipfs_cids': [
            'bafybeia3telu2s742xco3ap5huh4tk45cikwuxczwhrd6gwc3rcuat7odq',
        ],
        'sha256': 'ef352b60f2577692ab3e9da19d09a49fa9da9937f892afc48094988a17c32dc3',
        'sha512': '80f3f5a5bf5b225c36cbefe44e0c977bf9f3ea53658a97bc2d215405587f40dea6b6c0f04b5934129b4c0265616846562c3f15c9aba61ae1afaacd13c047c9cb',
        'properties': {
            'shape': (
                3000,
                3836,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'astro': {
        'fname': 'astro.png',
        'url': 'https://i.imgur.com/KXhKM72.png',
        'note': 'An image of Eileen Collins.',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb78a71cc6eae69303a7/download',
        ],
        'ipfs_cids': [
            'bafybeif2w42xgi6vkfuuwmn3c6apyetl56fukkj6wnfgzcbsrpocciuv3i',
        ],
        'sha256': '9f2b4671e868fd51451f03809a694006425eee64ad472f7065da04079be60c53',
        'sha512': 'de64fcb37e67d5b5946ee45eb659436b446a9a23ac5aefb6f3cce53e58a682a0828f5e8435cf7bd584358760d59915eb6e37a1b69ca34a78f3d511e6ebdad6fd',
        'properties': {
            'shape': (
                512,
                512,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'carl': {
        'fname': 'carl.jpg',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/b/be/Carl_Sagan_Planetary_Society.JPG',
        'mirrors': [
            'https://i.imgur.com/YnrLyry.jpg',
            'https://data.kitware.com/api/v1/file/647cfb8da71cc6eae69303b0/download',
        ],
        'note': 'An image of Carl Sagan.',
        'ipfs_cids': [
            'bafkreieuyks2z7stz56q63dvk555sr57kwnevgoruiaob7ffg5qcvftnui',
        ],
        'sha256': '94c2a5acfe53cf7d0f6c75577bd947bf559a4a99d1a200e0fca537602a966da2',
        'sha512': 'dc948163225157b85a968b2614cf2a2416b98d8b7b115ce8e046744e64e0f01150e539c06e78fc58306725188ee84f443414abac2e95dc11a8f2435df97ab6d4',
        'properties': {
            'shape': (
                448,
                328,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'lowcontrast': {
        'fname': 'lowcontrast.jpg',
        'url': 'https://i.imgur.com/dyC68Bi.jpg',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb93a71cc6eae69303b3/download',
        ],
        'ipfs_cids': [
            'bafkreictevzkeroswqavqneizt47am7fsyg4t47vnogknojtvcmg5spjly',
        ],
        'sha256': '532572a245d2b401583488ccf9f033e5960dc9f3f56b8ca6b933a8986ec9e95e',
        'sha512': '68d37c11a005168791e6a6ca018d34c6ee305c76a38fa8c93ccfaf4520f2f01d690b218b4ad6fbac36790104a670a154daa2da14850b5de0cc7c5d6843e5b18a',
        'properties': {
            'shape': (
                267,
                400,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 85,
            'max_value': 193,
        },
    },
    'paraview': {
        'fname': 'paraview.png',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/46/ParaView_splash1.png',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb97a71cc6eae69303b6/download',
        ],
        'ipfs_cids': [
            'bafkreiefsqr257hban5sw2kzw5gxwe32ieckzvw3swusi6v3e3bnbkutxa',
        ],
        'sha256': '859423aefce1037b2b6959b74d7b137a4104acd6db95a9247abb26c2d0aa93b8',
        'sha512': '25e92fe7661c0d9caf8eb919f6a9e76ed1bc689b1c599ad0786a47b86578961b07746a8303deb9efdab2bb562c700751d8cf6555e628bb65cb7ea74e8da8ad23',
        'properties': {
            'shape': (
                106,
                462,
                4,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'parrot': {
        'fname': 'parrot.png',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb9ca71cc6eae69303b9/download',
        ],
        'ipfs_cids': [
            'bafkreih23vgn3xcg4qyylgmueholdlu5hotnco23nufmybjgr7dsi3z6le',
        ],
        'sha256': 'fadd4cdddc46e43185999421dcb1ae9d3ba6d13b5b6d0acc05268fc7246f3e59',
        'sha512': '542f08ae6228483aa418ed1108d99a63805292bae43388256ea3edad780f7de2654ace72efcea4259b44a41784c364543fe763d4e4c65c90221be4b70e2d056c',
        'properties': {
            'shape': (
                200,
                150,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'stars': {
        'fname': 'stars.png',
        'url': 'https://i.imgur.com/kCi7C1r.png',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfba7a71cc6eae69303bf/download',
        ],
        'ipfs_cids': [
            'bafkreibwhenu2nvuwxrfs7ct7fdfsumravbpx3ec6wqccnyvowor32lrj4',
        ],
        'sha256': '36391b4d36b4b5e2597c53f9465951910542fbec82f5a0213715759d1de9714f',
        'sha512': 'e19e0c0c28c67441700cf272cb6ae20e5cc0baee24e5527e096e61e290ca823913224cdbabb884c5550e73587192428c0650921a00630c82f45c4eddf52c652f',
        'properties': {
            'shape': (
                256,
                256,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'pm5644': {
        'fname': 'Philips_Pattern_pm5644.png',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/47/Philips_Pattern_pm5644.png',
        'note': 'A test pattern good for checking aliasing effects',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfba2a71cc6eae69303bc/download',
        ],
        'ipfs_cids': [
            'bafkreihluuadifesmsou7jhihnjabk577jthhxs54tba5vtj33pladjzie',
        ],
        'sha256': 'eba500341492649d4fa4e83b5200abbffa6673de5de4c20ed669dedeb00d3941',
        'sha512': '8841ccd59b41dde98385e93531837668f09fafa42cfbdf27bf7c1088028596e3c82da8cad102543b330e1bba97476060ce002864360da76b2b3116647d2a79d8',
        'properties': {
            'shape': (
                576,
                1024,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'tsukuba_l': {
        'fname': 'tsukuba_l.png',
        'url': 'https://i.imgur.com/DhIKgGx.png',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfbaba71cc6eae69303c2/download',
            'https://raw.githubusercontent.com/tohojo/image-processing/master/test-images/middlebury-stereo-pairs/tsukuba/imL.png',
        ],
        'ipfs_cids': [
            'bafkreihcsfciih2oeiaordvwvwjiz6r64dcvzswaukctfsjjhrff4cziju',
        ],
        'sha256': 'e29144841f4e2200e88eb6ad928cfa3ee0c55ccac0a28532c9293c4a5e0b284d',
        'sha512': '51b8df8fb08f12609676923bb473c76b8ef9d73ce2c5493bca00b7b4b0eec7b298ce33f0bf860cc94c8b7cda8e69e021674e5a7ddaf0a1f007318053e4985740',
        'properties': {
            'shape': (
                288,
                384,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'tsukuba_r': {
        'fname': 'tsukuba_r.png',
        'url': 'https://i.imgur.com/38RST9H.png',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfbb0a71cc6eae69303c5/download',
            'https://raw.githubusercontent.com/tohojo/image-processing/master/test-images/middlebury-stereo-pairs/tsukuba/imR.png',
        ],
        'ipfs_cids': [
            'bafkreih3j2frkyobo6u2xirwso6vo3ioa32xpc4nitq4dtc4lxjv2x6r2q',
        ],
        'sha256': 'fb4e8b1561c177a9aba23693bd576d0e06f5778b8d44e1c1cc5c5dd35d5fd1d4',
        'sha512': '04da24efa0037aaad7a72a19d2210dd64f39f1a703d12fd1b379c3d6a9fb8695f33584d566b6159eb9aebce5b9b930b52df4b2ae7e90fcf66014711063635c27',
        'properties': {
            'shape': (
                288,
                384,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
}


def grab_test_image_fpath(key='astro', dsize=None, overviews=None, allow_fallback=True):
    """
    Ensures that the test image exists (this might use the network) and returns
    the cached filepath to the requested image.

    Args:
        key (str): which test image to grab. Valid choices are:
            astro - an astronaught
            carl - Carl Sagan
            paraview - ParaView logo
            stars - picture of stars in the sky
            OR can be an existing path to an image

        dsize (None | Tuple[int, int]):
            if specified, we will return a variant of the data with the
            specific dsize

        overviews (None | int):
            if specified, will return a variant of the data with overviews

        allow_fallback (bool):
            if True, and for some reason (e.g. network issue) we cannot grab
            the requested image, generate a random image based with expected
            metadata.

    Returns:
        str: path to the requested image
    """
    try:
        item = _TEST_IMAGES[key]
    except KeyError:
        valid_keys = sorted(_TEST_IMAGES.keys())
        cand = ub.Path(key)
        if cand.exists() and cand.is_file():
            return cand
        else:
            raise KeyError(
                'Unknown key={!r}. Valid keys are {!r}'.format(
                    key, valid_keys))
    if not isinstance(item, dict):
        item = {'url': item}

    grabkw = {
        'appname': 'kwimage/demodata',
    }
    hasher_priority = ['sha256']
    for hasher in hasher_priority:
        if hasher in item:
            grabkw.update({
                'hash_prefix': item[hasher],
                'hasher': hasher,
            })
            break
    if 'fname' in item:
        grabkw['fname'] = item['fname']

    ipfs_gateways = [
        'https://ipfs.io/ipfs',
        'https://dweb.link/ipfs',
        # 'https://gateway.pinata.cloud/ipfs',
    ]
    url = item['url']
    mirror_urls = []
    if 'mirrors' in item:
        mirror_urls += item['mirrors']
    if 'ipfs_cids' in item:
        for cid in item['ipfs_cids']:
            for gateway in ipfs_gateways:
                ipfs_url = gateway + '/' + cid
                mirror_urls.append(ipfs_url)

    try:
        fpath = _grabdata_with_mirrors(url, mirror_urls, grabkw)
    except Exception:
        if allow_fallback:
            # To avoid network issues in testing, add an option that triggers
            # if all mirrors fail. In that case, create a random image according
            # to the specs. Ideally use a different path, so if networking comes
            # back on we get the real image if we can.
            import numpy as np
            cache_dpath = ub.Path.appdir(grabkw['appname'])
            fname = ub.Path(item['fname']).augment(stemsuffix='_random_fallback')
            fallback_fpath = cache_dpath / fname
            if not fallback_fpath.exists():
                shape = item['properties']['shape']
                # dtype = item['properties']['dtype']
                min_value = item['properties']['min_value']
                max_value = item['properties']['max_value']
                rand_data = np.random.rand(*shape)
                rand_data = (rand_data * (max_value - min_value)) + min_value
                rand_data = rand_data.astype(item['properties']['dtype'])
                _imwrite_png(fallback_fpath, rand_data)
            return fallback_fpath
        else:
            raise

    augment_params = {
        'dsize': dsize,
        'overviews': overviews,
    }
    for k, v in list(augment_params.items()):
        if v is None:
            augment_params.pop(k)

    if augment_params:
        raise NotImplementedError
    return fpath


def _imwrite_png(fpath, data):
    """
    For fallbacks, if all other image writers are not available, we ship with a
    restricted small pure-python png writter.

    Args:
        fpath (str): path to write to
        data (ndarray): image as a numpy array

    References:
        https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image
        https://blender.stackexchange.com/questions/62072/does-blender-have-a-method-to-a-get-png-formatted-bytearray-for-an-image-via-pyt/62218#62218
        http://www.libpng.org/pub/png/spec/1.2/PNG-Chunks.html

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/utool'))
        from utool._kwimage_im_demodata import *  # NOQA
        from utool._kwimage_im_demodata import _grabdata_with_mirrors, _imwrite_png
        fpath = 'foo.png'
        shape = (5, 7, 4)
        data01 = np.linspace(0, 1, np.prod(shape)).reshape(*shape)
        data01 = np.random.rand(*shape)
        data = (data01 * 255).astype(np.uint8)
        _imwrite_png(fpath, data)

        import kwimage
        recon = kwimage.imread(fpath)

        import kwplot
        kwplot.autompl()
        kwplot.figure(fnum=1, doclf=1)
        kwplot.imshow(data, pnum=(1, 2, 1), fnum=1)
        kwplot.imshow(recon, pnum=(1, 2, 2), fnum=1)
        assert np.all(recon == data)
    """
    import zlib
    import struct
    import numpy as np

    assert data.dtype.kind == 'u'
    assert data.dtype.itemsize == 1

    height, width = data.shape[0:2]

    if len(data.shape) == 2:
        channels = 1
    else:
        assert len(data.shape) == 3
        channels = data.shape[2]

    assert channels in {1, 3, 4}
    data = np.ascontiguousarray(data)
    buf = bytearray(data.tobytes())

    # keep vertical line order add null bytes at the start
    pixel_byte_width = width * channels
    span_gen = list(range(0, (height) * pixel_byte_width, pixel_byte_width))
    raw_data = b''.join(
        b'\x00' + buf[span:span + pixel_byte_width]
        for span in span_gen
    )

    def png_pack(png_tag, data):
        chunk_head = png_tag + data
        return (struct.pack("!I", len(data)) +
                chunk_head +
                struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

    png_header = b'\x89PNG\r\n\x1a\n'
    bit_depth = 8

    if channels == 1:
        # grayscale sample
        color_type = 0
    elif channels == 2:
        # grayscale followed by alpha
        color_type = 4
    elif channels == 3:
        # rgb triple
        color_type = 2
    elif channels == 4:
        # rgb triple followed by alpha
        color_type = 6

    compression_method = 0
    filter_method = 0
    interlace_method = 0
    format_header = struct.pack(
        "!2I5B", width, height,
        bit_depth,
        color_type,
        compression_method,
        filter_method,
        interlace_method,
    )

    png_bytes = b''.join([
        png_header,
        png_pack(b'IHDR', format_header),
        png_pack(b'IDAT', zlib.compress(raw_data, 9)),
        png_pack(b'IEND', b'')])

    with open(fpath, 'wb') as file:
        file.write(png_bytes)
