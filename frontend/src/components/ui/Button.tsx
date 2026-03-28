interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md'
}

const base = 'inline-flex items-center justify-center rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/40'

const variantStyles: Record<string, string> = {
  primary: 'bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary-hover)]',
  secondary: 'bg-[var(--color-surface)] text-[var(--color-text)] border border-[var(--color-border)] hover:bg-[var(--color-surface-hover)]',
  ghost: 'text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-hover)]',
}

const sizeStyles: Record<string, string> = {
  sm: 'px-3 py-1.5 text-xs',
  md: 'px-4 py-2 text-sm',
}

export function Button({ variant = 'primary', size = 'md', className = '', ...props }: ButtonProps) {
  return (
    <button
      className={`${base} ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
      {...props}
    />
  )
}
