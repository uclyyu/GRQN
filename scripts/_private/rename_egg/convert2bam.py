from glob import glob
import subprocess, os


if __name__ == '__main__':

	for root in ['train/*.egg', 'test/*.egg']:
		eggfiles = glob(root)

		for eggfile in eggfiles:
			bamfile = eggfile.replace('.egg', '.bam')
			subprocess.call('egg2bam {} {}'.format(eggfile, bamfile).split(' '))
			print('{} ---> {}'.format(eggfile.split(os.path.sep)[-1], bamfile.split(os.path.sep)[-1]))
